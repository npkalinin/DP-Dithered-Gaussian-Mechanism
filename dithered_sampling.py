import mpmath as mp
import torch


def _sample_from_cdf_rows(x, gamma, bits_or_prefix, a, sigma, cdf_error, nbytes=None):
    if nbytes is None:
        bits = bits_or_prefix
        nbytes = bits.shape[1]
        shifts = 8 * torch.arange(nbytes - 1, -1, -1, device=bits.device, dtype=torch.int64)
        prefix = (bits.to(torch.int64) << shifts).sum(1)
    else:
        prefix = bits_or_prefix

    scale = 2.0 ** (-8 * nbytes)
    pf = prefix.to(torch.float64)
    u = (pf + 0.5) * scale
    q = (x + sigma * torch.special.ndtri(u)) / a - gamma - 0.5

    valid = torch.isfinite(q) & (torch.abs(q) < 2.0 ** 52)
    k = torch.ceil(torch.where(valid, q, 0.0)).to(torch.int64)

    kf = k.to(torch.float64)
    t = ((a * kf - x) + a * gamma) / sigma
    lo, hi = pf * scale, (pf + 1) * scale

    stable = torch.isfinite(t) & (torch.abs(kf) < 2.0 ** 50) & (torch.abs(x / a) < 2.0 ** 50)
    safe = valid & stable \
        & (lo >= torch.special.ndtr(t - 0.5 * a / sigma) + cdf_error) \
        & (hi <= torch.special.ndtr(t + 0.5 * a / sigma) - cdf_error)

    return prefix, k, safe


_sample_from_cdf_rows_compiled = torch.compile(_sample_from_cdf_rows)


def _sample_high_precision_row(x, a, gamma, sigma, prefix, nbytes, noise_fn,
                               min_prec=128, guard_bits=64):
    ratios = [float(v).as_integer_ratio() for v in (x, a, gamma, sigma)]
    prefix, nbytes = int(prefix), int(nbytes)
    k = None
    info = torch.iinfo(torch.int64)

    while True:
        prec = max(min_prec, 8 * nbytes + guard_bits)

        with mp.workprec(prec):
            x_mp, a_mp, gamma_mp, sigma_mp = (mp.mpf(n) / d for n, d in ratios)
            sqrt2, half = mp.sqrt(2), mp.mpf("0.5")
            lo = mp.mpf(prefix) / (1 << (8 * nbytes))
            hi = mp.mpf(prefix + 1) / (1 << (8 * nbytes))

            def F(j):
                t = (a_mp * (mp.mpf(j) + gamma_mp + half) - x_mp) / sigma_mp
                return half * mp.erfc(-t / sqrt2) if t <= 0 \
                    else 1 - half * mp.erfc(t / sqrt2)

            if k is None:
                z = sqrt2 * mp.erfinv(lo + hi - 1)
                k = int(mp.ceil((x_mp + sigma_mp * z) / a_mp - gamma_mp - half))

            left, right = F(k - 1), F(k)

            while hi <= left:
                k -= 1
                right, left = left, F(k - 1)

            while lo >= right:
                k += 1
                left, right = right, F(k)

            err = mp.power(2, -(prec - guard_bits))
            if lo >= left + err and hi <= right - err:
                if not info.min <= k <= info.max:
                    raise OverflowError(f"Sample {k} does not fit in torch.int64.")
                return k

        prefix = 256 * prefix + int(
            torch.as_tensor(noise_fn(1), dtype=torch.uint8).reshape(-1)[0].item()
        )
        nbytes += 1


@torch.no_grad()
def dithered_Gaussian_Mechanism(x, a, gamma, sigma, noise_fn, initial_nbytes=2,
                                cdf_error=1e-12, max_fast_nbytes=6):
    device = x.device
    x = x.to(torch.float64)
    gamma = torch.as_tensor(gamma, device=device, dtype=torch.float64)
    a = torch.as_tensor(a, device=device, dtype=torch.float64)
    sigma = torch.as_tensor(sigma, device=device, dtype=torch.float64)

    d = x.numel()
    bits = torch.as_tensor(
        noise_fn(d * initial_nbytes), device=device, dtype=torch.uint8
    ).reshape(d, initial_nbytes)

    # Dense compiled pass: almost every coordinate should finish here.
    prefix, sample, safe = _sample_from_cdf_rows_compiled(
        x, gamma, bits, a, sigma, cdf_error
    )
    active = torch.where(~safe)[0]
    nbytes = initial_nbytes

    # Only the small unresolved subset is processed from here on.
    while active.numel() and nbytes < max_fast_nbytes:
        prefix[active] = 256 * prefix[active] + torch.as_tensor(
            noise_fn(active.numel()), device=device, dtype=torch.uint8
        ).reshape(-1).to(torch.int64)

        nbytes += 1
        _, k, safe = _sample_from_cdf_rows(
            x[active], gamma[active], prefix[active], a, sigma, cdf_error, nbytes
        )
        sample[active] = k
        active = active[~safe]

    a_hp, sigma_hp = a.item(), sigma.item()
    for idx in active.cpu().tolist():
        sample[idx] = _sample_high_precision_row(
            x[idx].item(), a_hp, gamma[idx].item(), sigma_hp,
            prefix[idx].item(), nbytes, noise_fn,
        )

    return sample
