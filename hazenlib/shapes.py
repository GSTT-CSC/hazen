"""
Shape classes used for analysis
"""


class Rod:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Rod: {self.x}, {self.y}'

    def __str__(self):
        return f'Rod: {self.x}, {self.y}'

    @property
    def centroid(self):
        return self.x, self.y

    def __lt__(self, other):
        """Using "reading order" in a coordinate system where 0,0 is bottom left"""
        try:
            x0, y0 = self.centroid
            x1, y1 = other.centroid
            return (-y0, x0) < (-y1, x1)
        except AttributeError:
            return NotImplemented

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
