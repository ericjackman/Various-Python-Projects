import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import *
import turtle as tur


# Creates canvas to store turtle screen
def create_canvas(root):
	canvas = tk.Canvas(root)
	canvas.config(width=850, height=600)
	canvas.pack(side=tk.BOTTOM)
	return canvas


# Creates user interface
def create_ui(root):
	row1 = Frame(root)
	row2 = Frame(root)
	row1.pack(side=tk.TOP)
	row2.pack(side=tk.TOP)

	button = tk.Button(row1, text='Build Fractal', command=fractal)
	button.pack(side=tk.LEFT, padx=5, pady=5)

	button2 = tk.Button(row2, text='Clear', command=lambda: reset(t))
	button2.pack(side=tk.LEFT, padx=5, pady=5)

	# Create combobox for selecting fractal type
	label = ttk.Label(row1, text='Fractal type:')  # label for combobox
	label.pack(side=tk.LEFT, padx=5, pady=5)

	types = tk.StringVar()
	type_pick = ttk.Combobox(row1, textvariable=types)
	type_pick.set('Tree')
	type_pick['values'] = ['Tree', 'Snowflake', 'Star']
	type_pick['state'] = 'readonly'
	type_pick.pack(side=tk.LEFT, padx=5, pady=5)

	def set_type(event):
		global model_type
		model_type = types.get()

	type_pick.bind('<<ComboboxSelected>>', set_type)

	color_choices = ['Black', 'White', 'Purple', 'Blue', 'Cyan', 'Green', 'Lime', 'Yellow', 'Orange', 'Brown',
					 'Red', 'Maroon', 'Pink']  # Options for colors

	# Create combobox for selecting color
	label = ttk.Label(row1, text='Pen color:')  # label for combobox
	label.pack(side=tk.LEFT, padx=5, pady=5)

	color = tk.StringVar()
	color_pick = ttk.Combobox(row1, textvariable=color)
	color_pick.set('Black')
	color_pick['values'] = color_choices
	color_pick['state'] = 'readonly'
	color_pick.pack(side=tk.LEFT, padx=5, pady=5)

	def set_color(event):
		t.color(color.get())

	color_pick.bind('<<ComboboxSelected>>', set_color)

	# Create combobox for selecting background color
	label2 = ttk.Label(row1, text='Background color:')  # label for combobox
	label2.pack(side=tk.LEFT, padx=5, pady=5)

	color2 = tk.StringVar()
	color_pick2 = ttk.Combobox(row1, textvariable=color2)
	color_pick2.set('White')
	color_pick2['values'] = color_choices
	color_pick2['state'] = 'readonly'
	color_pick2.pack(side=tk.LEFT, padx=5, pady=5)

	def set_bg_color(event):
		screen.bgcolor(color2.get())

	color_pick2.bind('<<ComboboxSelected>>', set_bg_color)

	# Create combobox for selecting number of iterations
	label3 = ttk.Label(row2, text='Number of iterations:')  # label for combobox
	label3.pack(side=tk.LEFT, padx=5, pady=5)
	iters = tk.IntVar()
	iter_pick = ttk.Combobox(row2, textvariable=iters)
	iter_pick.set(3)
	iter_pick['values'] = [1, 2, 3, 4, 5, 6, 7]
	iter_pick['state'] = 'readonly'
	iter_pick.pack(side=tk.LEFT, padx=5, pady=5)

	def set_iterations(event):
		global num_iter
		num_iter = iters.get()

	iter_pick.bind('<<ComboboxSelected>>', set_iterations)

	# Create combobox for selecting animation speed
	label4 = ttk.Label(row2, text='Animation Speed:')  # label for combobox
	label4.pack(side=tk.LEFT, padx=5, pady=5)
	speeds = tk.IntVar()
	speed_pick = ttk.Combobox(row2, textvariable=speeds)
	speed_pick.set(5)
	speed_pick['values'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	speed_pick['state'] = 'readonly'
	speed_pick.pack(side=tk.LEFT, padx=5, pady=5)

	def set_speed(event):
		if speeds.get() == 10:
			t.speed(0)
		else:
			t.speed(speeds.get())

	speed_pick.bind('<<ComboboxSelected>>', set_speed)

	# Create text input for size
	label5 = ttk.Label(row2, text='Size:')  # label for combobox
	label5.pack(side=tk.LEFT, padx=5, pady=5)
	size_text = tk.Text(row2, height=1, width=5)
	size_text.insert('1.0', '200')
	size_text.pack(side=tk.LEFT, padx=5, pady=5)

	def set_size():
		global size
		txt = size_text.get('1.0', 'end-1c')
		try:
			num = int(txt)
			if 50 <= num <= 300:
				size = num
			else:
				messagebox.showwarning('Warning', 'Size must be 50 - 300')
		except:
			messagebox.showwarning('Warning', 'Size must be 50 - 300')

	# Set button for size
	button3 = tk.Button(row2, text='Set', command=set_size)
	button3.pack(side=tk.LEFT, padx=5, pady=5)

	# Create text input for angle
	label6 = ttk.Label(row2, text='Angle:')  # label for combobox
	label6.pack(side=tk.LEFT, padx=5, pady=5)
	angle_text = tk.Text(row2, height=1, width=5)
	angle_text.insert('1.0', '90')
	angle_text.pack(side=tk.LEFT, padx=5, pady=5)

	def set_angle():
		txt = angle_text.get('1.0', 'end-1c')
		try:
			num = int(txt)
			if 0 <= num <= 360:
				t.setheading(num)
			else:
				messagebox.showwarning('Warning', 'Angle must be 0 - 360')
		except:
			messagebox.showwarning('Warning', 'Angle must be 0 - 360')

	# Set button for angle
	button4 = tk.Button(row2, text='Set', command=set_angle)
	button4.pack(side=tk.LEFT, padx=5, pady=5)


# Moves turtle to the mouse position
def set_position(canvas):
	x = canvas.winfo_pointerx() - canvas.winfo_rootx()  # find x of mouse in canvas
	y = canvas.winfo_pointery() - canvas.winfo_rooty()  # find y of mouse in canvas
	t.penup()
	t.goto(x - 425, 300 - y)  # move turtle to point (adjusting for different coordinate system)
	t.pendown()


# Clears canvas and returns turtle to starting position
def reset(turtle):
	turtle.clear()
	turtle.penup()
	turtle.home()
	turtle.pendown()
	t.setheading(90)


# Helper method for recursive fractal methods
def fractal():
	global model_type
	global num_iter
	global size
	if model_type == 'Tree':
		build_tree(t, size / 4, 0.9, 30, num_iter)
	elif model_type == 'Snowflake':
		for i in range(3):
			snowflake(t, size, 3, 60, num_iter)
			t.right(120)
	elif model_type == 'Star':
		star(t, size, num_iter)


# Recursive method for building tree fractal
def build_tree(turtle, branch_length, shorten_by, angle, iterations):
	if iterations > 0:
		turtle.forward(branch_length)
		new_length = branch_length * shorten_by
		turtle.left(angle)
		build_tree(turtle, new_length, shorten_by, angle, iterations - 1)
		turtle.right(angle * 2)
		build_tree(turtle, new_length, shorten_by, angle, iterations - 1)
		turtle.left(angle)
		turtle.backward(branch_length)


# Recursive method for building a snowflake fractal
def snowflake(turtle, length, shortening_factor, angle, iterations):
	if iterations == 0:
		turtle.forward(length)
	else:
		iterations = iterations - 1
		length = length / shortening_factor
		snowflake(turtle, length, shortening_factor, angle, iterations)
		turtle.left(angle)
		snowflake(turtle, length, shortening_factor, angle, iterations)
		turtle.right(angle * 2)
		snowflake(turtle, length, shortening_factor, angle, iterations)
		turtle.left(angle)
		snowflake(turtle, length, shortening_factor, angle, iterations)


# Recursive method for building star fractal
def star(turtle, length, iterations):
	if iterations == 0:
		return
	else:
		for i in range(5):
			turtle.forward(length)
			star(turtle, length / 3, iterations - 1)
			turtle.left(216)


if __name__ == '__main__':
	rt = tk.Tk()  # create tkinter window
	rt.title('Create Art Using Recursion')
	c = create_canvas(rt)  # store canvas as c

	screen = tur.TurtleScreen(c)  # create turtle screen in canvas
	t = tur.RawTurtle(screen, shape='classic')  # create turtle

	# Set default for user controllable variables
	model_type = 'Tree'
	t.color('black')
	num_iter = 3
	t.speed(5)
	size = 200
	t.setheading(90)

	# Event for mouse click
	c.bind("<Button 1>", lambda x: set_position(c))

	create_ui(rt)  # create ui

	rt.mainloop()  # show tkinter window
