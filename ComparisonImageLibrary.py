from PIL import Image
from itertools import izip
import cv2
import numpy
import os

IMAGE_FIXTURES = os.path.join(os.path.dirname(__file__), '../../tests/image_fixture/')


class DIFFERENCE(object):
    PERFECT = 0.09
    GOOD = 0.05
    BAD = 0.15
    TERRIBLE = 0.3


class ComparisonImageLibrary(object):

    def __init__(self, path=None, image=None):
        assert path or image
        self._path = path
        self._image = image.convert('RGB') if image else None

    @property
    def image(self):
        if self._image is None:
            self._image = Image.open(self._path).convert('RGB')
        return self._image

    @property
    def width(self):
        return self.image.width

    @property
    def height(self):
        return self.image.height

    def to_bytes(self):
        """
        Converts the current image to a particular format and returns it as a string that you can then access as an
        iterable of binary bytes.

        >>> kitten_0 = ComparisonImageLibrary(os.path.join(IMAGE_FIXTURES, 'kitten_0.png'))
        >>> len(kitten_0.to_bytes())
        205200
        """
        return self.image.tostring()

    def pixel(self, x, y):
        """
        Get pixel at point.

        @param x Horizontal pixel, starting with 0 at left
        @param y Vertical pixel, starting with 0 at top
        @returns (r, g, b)

        >>> kitten_0 = ComparisonImageLibrary(os.path.join(IMAGE_FIXTURES, 'kitten_0.png'))
        >>> kitten_0.pixel(50, 50)
        (84, 98, 85)
        """
        return self.image.getpixel((x, y))

    def crop(self, rect):
        """
        Creates a new MonkeyImage object from a rectangular selection of the current image.

        Arguments
        rect	  A tuple (x, y, w, h) specifying the selection. x and y specify the 0-based pixel position of the upper left-hand
                  corner of the selection. w specifies the width of the region, and h specifies its height, both in units of
                  pixels.
                    The image's orientation is the same as the screen orientation at the time the screenshot was made.

        Returns
        A new MonkeyImage object containing the selection.

        >>> kitten_0 = ComparisonImageLibrary(os.path.join(IMAGE_FIXTURES, 'kitten_0.png'))
        >>> kitten_0.pixel(50, 50)
        (84, 98, 85)
        >>> kitten_0 = ComparisonImageLibrary(os.path.join(IMAGE_FIXTURES, 'kitten_0.png'))
        >>> kitten_0.crop((50, 50, 25, 25)).pixel(0, 0)
        (84, 98, 85)
        """
        x, y, w, h = rect
        return ComparisonImageLibrary(image=self.image.crop((x, y, w + x, y + h)))

    def same_as(self, other, percent=DIFFERENCE.PERFECT):
        """
        Compares this MonkeyImage object to another and returns the result of the comparison.
        The percent argument specifies the percentage difference that is allowed for the two images to be "equal".

        >>> kitten_0 = ComparisonImageLibrary(os.path.join(IMAGE_FIXTURES, 'kitten_0.png'))
        >>> kitten_1 = ComparisonImageLibrary(os.path.join(IMAGE_FIXTURES, 'kitten_1.png'))
        >>> kitten_0.same_as(kitten_0)
        True
        >>> kitten_0.same_as(kitten_1)
        False
        """
        return self.difference(other) <= percent

    def difference(self, other):
        """
        Compares difference to other image.

        >>> kitten_0 = ComparisonImageLibrary(os.path.join(IMAGE_FIXTURES, 'kitten_0.png'))
        >>> kitten_0.difference(kitten_0)
        0.0
        >>> kitten_1 = ComparisonImageLibrary(os.path.join(IMAGE_FIXTURES, 'kitten_1.png'))
        >>> round(kitten_0.difference(kitten_1), 4)
        0.3092
        """
        i1 = self.image
        i2 = other.image

        pairs = izip(i1.getdata(), i2.getdata())
        if len(i1.getbands()) == 1:
            # for gray-scale JPEGs
            dif = sum(abs(p1 - p2) for p1, p2 in pairs)
        else:
            dif = sum(abs(c1 - c2) for p1, p2 in pairs for c1, c2 in zip(p1, p2))

        n_components = i1.size[0] * i1.size[1] * 3
        return ((dif / 255.0) / n_components)

    def part_of(self, other, threshold=DIFFERENCE.PERFECT):
        """
        Checks if this MoneyImage is a sub image of another and returns the location and percentage of match

        >>> kitten_0 = ComparisonImageLibrary(os.path.join(IMAGE_FIXTURES, 'kitten_0.png'))
        >>> sub_kitty_1 = ComparisonImageLibrary(os.path.join(IMAGE_FIXTURES, 'sub_kitten.png'))
        >>> sub_kitty_1.part_of(kitten_0)
        (26, 23)
        >>> not_kitten = ComparisonImageLibrary(os.path.join(IMAGE_FIXTURES, 'radio.png'))
        >>> not_kitten.part_of(kitten_0)
        False
        """

        haystack = numpy.array(self.image)
        needle = numpy.array(other.image)

        result = cv2.matchTemplate(haystack, needle, cv2.TM_CCORR_NORMED)

        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        if maxVal < threshold:       #i did maxVal/10 because maxval returns. like .313 and DIFFERENCE is .05.. soo you see?
            return None
        return maxLoc

    def save(self, path, format='PNG'):
        """
        Writes the current image to the file specified by filename, in the format specified by format.

        :param path:
        :param format:
        :return:

        >>> import tempfile
        >>> temppath = tempfile.mkdtemp()
        >>> tempfile = os.path.join(temppath, 'saved_image.png')
        >>> kitten_0 = ComparisonImageLibrary(os.path.join(IMAGE_FIXTURES, 'kitten_0.png'))
        >>> kitten_0.save(tempfile)
        >>> temp_kitten = ComparisonImageLibrary(path=tempfile)
        >>> kitten_0.same_as(temp_kitten)
        True
        """
        self.image.save(path, format)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
