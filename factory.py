from abc import ABCMeta, abstractmethod


class FactoryClass:
	def create_object(self, obj_type):
		if obj_type == 'a':
			return ClassA()
		elif obj_type == 'b':
			return ClassB()
		elif obj_type == 'c':
			return ClassC()
		return None


class IClass(metaclass=ABCMeta):
	@abstractmethod
	def create_object(self):
		return


class ClassA(IClass):
	def __init__(self):
		self.name = "ConcreteProductA"

	def create_object(self):
		return self


class ClassB(IClass):
	def __init__(self):
		self.name = "ConcreteProductB"

	def create_object(self):
		return self


class ClassC(IClass):
	def __init__(self):
		self.name = "ConcreteProductC"

	def create_object(self):
		return self


PRODUCT = FactoryClass().create_object('b')
print(PRODUCT.name)
