class Visitable(object):
	def accept(self, visitor): pass


class Visitor(object):
	def visit(self, element): pass


class BookStoreVisitor(Visitor):
	def __init__(self):
		self.totalPrice = 0.0

	def visitCD(self, cd):
		self.totalPrice += cd.price

	def visitBook(self, book):
		self.totalPrice += book.price

	def visit(self, items):
		for i in items:
			i.accept(self)


class CD(Visitable):
	def __init__(self, price):
		self.price = price

	def accept(self, visitor):
		visitor.visitCD(self)


class Book(Visitable):
	def __init__(self, price):
		self.price = price

	def accept(self, visitor):
		visitor.visitBook(self)


items = [Book(15), CD(20)]
visitor = BookStoreVisitor()
visitor.visit(items)
print(visitor.totalPrice)
