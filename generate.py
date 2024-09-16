import numpy
import cv2 as opencv


# TODO generate diagonals for testing
# pass in array_like for colors (even if they're 1-channel)
def gradient(first, second, steps=128, direction="down"):
    if direction == "down":
        i, f = first, second
        vert = True
    elif direction == "right":
        i, f = first, second
        vert = False
    elif direction == "up":
        i, f = second, first
        vert = True
    elif direction == "left":
        i, f = second, first
        vert = False
    else:
        raise NotImplementedError
    # gradient strip
    grad = numpy.linspace(i, f, num=steps, dtype=numpy.uint8)
    # squarify output image
    if vert:
        return numpy.tile(grad.reshape(grad.shape[0], 1, grad.shape[1]), (1, steps, 1))
    else:
        return numpy.tile(grad.reshape(1, grad.shape[0], grad.shape[1]), (steps, 1, 1))


if __name__ == "__main__":
    black = numpy.array([0,0,0])
    white = numpy.array([255,255,255])
    opencv.imwrite("tests/bwdowngrad.png", gradient(black, white, steps=256))
    # create and save other gradients as needed for testing
