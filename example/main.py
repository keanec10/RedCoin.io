#region modules					# use "region" and "endregion" directives to segment code so it's foldable in IDEs

from lib1 import hx1, hx2		# specify imported modules at the top of the script
from lib2 import kx1

#endregion modules

#region constants

MAX_NUM_ITERATIONS = 100		# constants are UPPER_SNAKE_CASE
GROUP_NAME = "Group 36"
OBJ1 = -1
OBJ2 = 0
OBJ3 = 1

#endregion constants

#region methods

#region sample					# logically split methods into different sectoins grouped by functionality

# fx - brief description of what the function does...
# @a: brief description of what parameter "a" is for...
# @b: message to print on each iteration
# @c: the name of the segment
# @returns: the name of the segment
def fx(a, b, c) :				# spaces between parameters and before ":", leave a line for spacing after the declaration
	
	i = 0
	while (i < a) :				# leave a line for spacing after the declaration, as with the function declaration
		
		print(b)
								# leave a line for spacing after the loop is completed
	print("Done with " + c)
	return c
								# leave two lines for spacing after function end

# gx_2 - reports the sign of a given value
# @x: value to compare
# @y: value to print when equality is found, defaults to 1
# @returns: nothing
def gx_2(x, y = 1) :
									# try to keep local comments aligned where possible
	val = ((x + y) * x)				# better to use too many brackets than not enough brackets
	sample_msg = "Reporting: "		# use snake_case in function & variable names instead of camelCase
	if (x > 0) :					# space after "if" and before ":", spaces in the condition but not beside "(" or ")"
		
		print("Greater than 0")
		
	elif (x == 0) :
		
		print(sample_msg + str(y))
		
	else :
		
		print("Less than 0")


#endregion sample

#endregion methods
