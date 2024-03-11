class Command(object):
	def __init__(self, editor):
		self.editor = editor
		self.snapshot = []

	def make_backup(self, momento):
		self.snapshot.append(momento)

	def undo(self):
		self.editor.restore(self.snapshot[-1])
		self.snapshot.pop()


class Editor(object):
	def __init__(self):
		self.state = 0

	def set_state(self, num):
		self.state = num

	def print_state(self):
		print('Current editor state is ' + str(self.state))

	def save(self):
		return Momento(self.state)

	def restore(self, momento):
		self.state = momento.state


class Momento(Editor):
	def __init__(self, state):
		self.state = state

	def get_state(self):
		return self.state



e = Editor()
e.set_state(1)
e.print_state()
s = e.save()
c = Command(e)
c.make_backup(s)
e.set_state(2)
e.print_state()
s = e.save()
c.make_backup(s)
e.set_state(3)
e.print_state()
c.undo()
e.print_state()
c.undo()
e.print_state()
