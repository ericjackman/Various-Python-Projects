import math


class RoundHole(object):
	def __init__(self, radius):
		self._radius = radius

	def get_radius(self):
		return self._radius

	def fits(self, peg):
		if self._radius > peg.get_radius():
			return True
		return False


class RoundPeg(object):
	def __init__(self, radius):
		self._radius = radius

	def get_radius(self):
		return self._radius


class SquarePeg(object):
	def __init__(self, width):
		self._width = width

	def get_width(self):
		return self._width


class SquarePegAdapter(RoundPeg):
	def __init__(self, squarepeg):
		self._squarepeg = squarepeg

	def get_width(self):
		return self._squarepeg.get_width()

	def get_radius(self):
		return math.sqrt((self._squarepeg.get_width()**2) * 2) / 2


round_hole = RoundHole(7.1)
round_peg = RoundPeg(7)
print(round_hole.fits(round_peg))

round_hole = RoundHole(7.1)
square_peg = SquarePeg(10)
square_adapter = SquarePegAdapter(square_peg)
print(f"square width {square_adapter.get_width()} => {square_adapter.get_radius()}")
print(round_hole.fits(square_adapter))
