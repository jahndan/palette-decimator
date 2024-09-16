from palettes import palettes

# import math
import numpy as np
import cv2 as opencv


# takes 2n, where n is an exponent of 2: returns 2^n * 2^n bayer matrix
def bayer(two_n: int):
    if two_n < 2 or not np.log2(two_n).is_integer():
        print(two_n, "is not a valid input!")
        raise ValueError
    elif two_n == 2:
        return 0.25 * np.array([[0, 2], [3, 1]])
    else:
        rec = (two_n * two_n) * bayer(two_n // 2)
        return (1 / (two_n * two_n)) * np.block(
            [
                [rec + 0, rec + 2],
                [rec + 3, rec + 1],
            ]
        )


# ideally also setup a thing to use OKLCH (OKLAB) color picking for perceptually uniform gradients
# also setup optimal palletization based on what colors are already in the image


def color_diff(v, w):
    assert v.shape[-1] == w.shape[-1]  # same number of channels
    if len(v.shape) > len(w.shape):  # either same dims or w is bigger
        return color_diff(w, v)
    axes = tuple(range(1, len(w.shape)))  # evaluate norm along these axes
    tiles = list(w.shape[:-1]) + [1]  # number of times to be tiled along each axis
    vv = np.tile(v, tiles)  # make vv so it has the same shape as w
    return np.linalg.vector_norm(vv - w, axis=axes)


# returns index for the nth closest color (n-indexed)
# pass in range(n) for the n closest colors' indices (not including the nth)
def n_closest(pixel, palette, n=0):
    return palette[np.argpartition(color_diff(pixel, palette), n)[n]]


# TODO optimize for parallelizing where possible
# maybe support "uniform" and optimized palettes separately
# assuming dtype=np.uint8
def remap_colors(img, palette, dither_strategy: str = "none"):
    assert len(img.shape) == 3  # 2d array of pixels (multi-channel colors)
    assert len(palette.shape) == 2  # array of multi-channel colors
    assert img.shape[-1] == palette.shape[-1]  # same number of channels

    im = np.zeros_like(img)  # img.copy()
    height, width = img.shape[:-1]
    rng = np.random.default_rng()  # rng source
    current = np.zeros_like(img[0])  # sized like a row of img
    forward = np.zeros_like(img[0])  # sized like a row of img

    # note: for consistency with other dither implementations, use integer arithmetic
    match dither_strategy.split("-"):
        case ["none"] | ["threshold"]:
            for i in range(height):
                for j in range(width):
                    # map to nearest in palette
                    im[i, j] = n_closest(img[i, j], palette)
        case ["uniform", "white", "noise"]:
            for i in range(height):
                for j in range(width):
                    # get two nearest colors in palette
                    first, second = n_closest(img[i, j], palette, range(2))
                    # compute scaled random offset
                    diff = np.clip(color_diff(first, second), -256, 256)
                    rand = rng.uniform(low=-0.5, high=0.5)
                    offset = np.int8(1.0 * diff * rand)
                    # map to nearest with offset applied
                    im[i, j] = n_closest(img[i, j] + offset, np.array([first, second]))
        case ["tri", "white", "noise"]:
            for i in range(height):
                for j in range(width):
                    # get two nearest colors in palette
                    first, second = n_closest(img[i, j], palette, range(2))
                    # compute scaled random offset
                    diff = np.clip(color_diff(first, second), -256, 256)
                    rand = rng.triangular(-0.5, 0.0, 0.5)
                    offset = np.int8(2.0 * diff * rand)
                    # map to nearest with offset applied
                    im[i, j] = n_closest(img[i, j] + offset, np.array([first, second]))
        case ["blue", "noise"]:
            # TODO precompute
            # TODO implement
            # TODO raise error if not precomputed
            raise NotImplementedError
        # TODO other, less optimal, colors of noise? for stylizing/experimenting purposes?
        case ["bayer", num] if num.isnumeric():
            n = int(num)
            mat = bayer(2 * n)  # takes 2n, where 2^n = num
            mat = np.tile(mat, (1 + height // n, 1 + width // n))
            for i in range(height):
                for j in range(width):
                    # get two nearest colors in palette
                    first, second = n_closest(img[i, j], palette, range(2))
                    # compute scaled random offset
                    diff = np.clip(color_diff(first, second), -256, 256)
                    rand = 2.0 * mat[i, j] - 1.0
                    offset = np.int8(1.5 * diff * rand)
                    # map to nearest with offset applied
                    im[i, j] = n_closest(img[i, j] + offset, np.array([first, second]))
        case ["sierra", "lite"]:
            # TODO implement
            raise NotImplementedError
        case ["floyd", "steinberg"]:
            # TODO implement
            raise NotImplementedError
        case ["burkes"]:
            # TODO implement
            raise NotImplementedError
        case ["sierra"]:
            # TODO implement
            raise NotImplementedError
        case _:
            print("Unknown dither strategy passed in to remap_colors()")
            raise NotImplementedError
    return im


# in theory, non-naive and naive scoring should perform similarly on not-dithered remaps
# dithered remaps *should* perform better on non-naive but this needs empirical testing
def evaluate_score(img, mapped, naive_scoring=True):
    assert len(img.shape) == 3

    mag_diff = lambda v, w: np.linalg.vector_norm(v - w, axis=(0, 1))
    if naive_scoring:
        score = lambda v, w: sum(mag_diff(v[1:2, 1:2], w[1:2, 1:2]))
    else:
        score = lambda v, w: sum(mag_diff(v, w)) / 9.0

    padding = ((1, 1), (1, 1), (0, 0))
    vimg = np.pad(img, padding)
    wimg = np.pad(mapped, padding)

    total_score = 0.0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            total_score += score(vimg[i : i + 2, j : j + 2], wimg[i : i + 2, j : j + 2])
    return total_score


def stall():
    while True:
        if (opencv.waitKey(1) & 0xFF) == ord("q"):
            break


if __name__ == "__main__":
    import sys

    palette = palettes[int(sys.argv[1])]

    for filename in sys.argv[2:]:
        img = opencv.imread(filename, opencv.IMREAD_UNCHANGED)
        # resized so it doesn't take so long
        if img.size > 3 * 1000000:
            scale = 3000000.0 / img.size
            t_img = opencv.resize(img, None, fx=scale, fy=scale)
        else:
            t_img = img

        uniwn_map = remap_colors(t_img, palette, dither_strategy="uniform-white-noise")
        # triwn_map = remap_colors(t_img, palette, dither_strategy="tri-white-noise")
        # bluen_map = remap_colors(t_img, palette, dither_strategy="blue-noise")
        bayer4_map = remap_colors(t_img, palette, dither_strategy="bayer-4")
        bayer8_map = remap_colors(t_img, palette, dither_strategy="bayer-8")
        bayer16_map = remap_colors(t_img, palette, dither_strategy="bayer-16")
        bayer32_map = remap_colors(t_img, palette, dither_strategy="bayer-32")

        print(
            f"{filename} {t_img.shape} [{t_img.size}]",
            f"control: {evaluate_score(t_img, t_img)}",
            f"uniform white noise: {evaluate_score(t_img, uniwn_map)}",
            # f"triangular white noise: {evaluate_score(t_img, triwn_map)}",
            # f"blue noise: {evaluate_score(t_img, bluen_map)}",
            f"bayer 4x4 ordered: {evaluate_score(t_img, bayer4_map)}",
            f"bayer 8x8 ordered: {evaluate_score(t_img, bayer8_map)}",
            f"bayer 16x16 ordered: {evaluate_score(t_img, bayer16_map)}",
            f"bayer 32x32 ordered: {evaluate_score(t_img, bayer32_map)}",
            sep="\n",
        )

        opencv.imshow(
            filename,
            np.block(
                [
                    [
                        [t_img],
                        [uniwn_map],
                        # [triwn_map],
                        # [bluen_map],
                        [bayer4_map],
                        [bayer8_map],
                        [bayer16_map],
                        [bayer32_map],
                    ],
                ]
            ),
        )
        opencv.waitKey(1)

    stall()
    opencv.destroyAllWindows()
