
# printing
print("float %3.3f, integer %d, string %s " %( 10.5, 23, 'hello'  ) )  # string can work for datetime as well
print(f" test {x} " )
x = "var1";   print(  "testing {x}...".format( x=x )  ) 

# Upgrade library.  pip install --upgrade xxxx  (--upgrade can be replace with -U)
# Uninstall library.  pip uninstall xxxx

# If some package installed didn't work, try installing with conda.
# "conda install <package> -c conda-forge

 # Enter debug mode.  ctrl+d will exit debug mode.  Use "if self.name=='worker_0':" if running parallel.
import pdb; pdb.set_trace()    

# Change windowspath to Linux path.
os.path.normpath(r"C:\xxx\yyy").replace(os.sep, '/')

# run unix command
os.system('ls')

# Create list in one line 
vec = [ x*2 for x in range(10) ]  
vec = [ *range(0,10,1) ]  # also work.   
[4 if x==1 else x for x in a]   # using if.
   
# Check class variables.  use  locals() or globals() to check list of variables
classX.__dict__.keys()  

# curl usage
# https://stackoverflow.com/questions/62326253/how-do-i-post-curl-command-as-a-python-request  ( for using "curl post" )

# Removes all log texts on console. 
os.system('cls')     

# To clear memory
# use "globals()" "del xxx" "gc.collect()" "tf.keras.backend.clear_session()" to clear memory. 

 # get unique list
set(['a','b','a'])    

# Convert dictionary to string with delimeter "__"
"__".join(["=".join([key, str(val)]) for key, val in dic.items()])  

# Upload files in google colab. For exporting, see https://stackoverflow.com/questions/53898836/export-dataframe-as-csv-file-from-google-colab-to-google-drive
from google.colab import files   files.upload()   

# If installing tensorflow fails due to "The system cannot find the file specified"
pip install --user tensorflow  

# Remove folder in python code.
import shutil   shutil.rmtree('./mnt')    

# This will create log in text file
python -m trace --trace YOURSCRIPT.py  > log.txt 2>&1    

# Improve python speed. Numba package to increase python code speed. 
from numba import jit, prange
@jit(nogil=True, parallel=True)
def parallel_function(data):
    for i in prange(len(data)):
        xxx = 0.0

# Find number of combinations m subsets, having n sets.  Binomial coefficient.  
n!/m!(n-m)!

# Extract array from random indices.
rand_ind = random.sample( range(num_data), k=int(num_data*0.8) )
non_rand_ind = list(set([*range(num_data)]) - set(rand_ind))

# use getopt.
import getopt
options, args = getopt.getopt( sys.argv[1:], 'd:c:', ['csv_data=', 'col_txt='])
# Run as  python xxx.py  -d <file1> -c <file2>,  or  python xxx.py  --csv_data=<file1>  --col_txt=<file2>

# To install IPython, and use it in jupyter notebook.
python -m ipykernel install --user --name <yourenvname> --display-name <display-name>

# If you want to show progress bar in for loop.
from tqdm.auto import tqdm  
for i in tqdm( xlist ):

# check if the variable type is certain type.
isinstance( x, int ) 

# Check several conditions.
if all(True, False, True):
if any(True, False, True):

# Get directory where you excuted the python code .py
from pathlib import Path
Path('__file__').resolve().parent

# For append(), remove(), use deque library.

# Python dash sytax.
python -i my_script.py  # debug
python -m timeit   # measure runtime

# Check if element exists in list.
x = ['a', 'b']
if 'a' in set(x):  if 'a' in {*x}:

# Download python library folder.
pip install <library_name> -t .

# Use einstein notation
np.einsum('ij -> jk', A, B)

# datetime parser
from dateutil import parser
dt = parser.parse("7:19 AM on March 9th")
df['col'].apply(lambda x: parser.parse(x))  # for pandas

#------------------------------------------------------------------------



#--------- to check bugs in your code -----------------------------------------------------------------
echo $?   # type this in anaconda power shell after the code stops.  If False, something went wrong.

# If you see unicode errors, insert below in the very first line of your script and trace.py
Get-content -tail  10000  log8.txt > log10000.txt   # This is actually powershell command.  
if os.name == 'nt':
    import _locale
    _locale._getdefaultlocale_backup = _locale._getdefaultlocale
    _locale._getdefaultlocale = (lambda *args: (_locale._getdefaultlocale_backup()[0], 'UTF-8'))
    
with open( 'download_list1.csv', 'r', encoding='utf-8' ) as ff:  # if you see encording errors, use encoding='utf-8-sig'
#-------------------------------------------------------------------------------------------------------

#---------- For string match search ---------------
import fnmatch  # for string match (like wildcard)   if fnmatch.fnmatch(string, '*xxxx*'):
import re
x = "xxxxxAAA12345ZZZxxxxxxxx"
re.search('AAA(.+?)ZZZ', x).group(1)
y = "xxxx[[12345]]xxxxxx"
re.search('\[\[(.+?)\]\]', y).group(1)  # need backslash before brackets, parenthesis.  
#--------------------------------------------------

#---- append value to file ------------------------
f= open("C:/Users/ken_nakatsukasa/Desktop/BitBucket1968_test/loss.txt","a")
f.write("%6.6f\n" % (loss.numpy() )  )
f.close()
#--------------------------------------------------

#------- write list to text file ---------------------------------
final = ( new_model['tech'],  str(upper_max), str(lower_max) )
f = open('../result.txt', 'w')
f.write(" ".join(final) )
f.close()
#----------------------------------------------------------------- 

#-------------- system related -----------------------------------
import sys
import os

# adding system path
sys.path.append('C:/folder_ken/mai-mate/mai-mate2-ai-data-job/src')

# check if variable exists or not
is_local = "a_variable" in locals()   # use global()  instead of locals() if defined globally

# change directory
os.chdir('./tmp')

# list all files in directory
files = os.listdir(path='/tmp')  

# run cmd command 
os.system('cp xxx.py ./yyy.py')

# if file/dir exits or not
os.path.isdir('./xxx')   # or use os.path.exists('./xxx.txt')  for file  
os.makedirs(os.path.join(current_dir, new_dir))

# set environment variable
os.environ['S3_BUCKET_NAME'] = 'invast-mai-stg-mai-mate2-ai-engine-model-by-user'
os.environ.get("S3_BUCKET_NAME")

# To examine files in the directory.
for dirpath, dirnames, filenames in os.walk("pizza_steak"):

# To get list of files that are sorted in created time.
win_files = sorted( Path().iterdir(), key=os.path.getmtime )

#------------------------------------------------------------------

#------------- datetime -------------------------------------------
import datetime as dt
#datetime writing
x = dt.datetime(2020,9,27,12,0,0,0)
y = dt.datetime.strptime(str(2020-08-28 19:59:00+00:00).replace('+00:00',''), '%Y-%m-%d %H:%M:%S')  # string to datetime

z = x + dt.timedelta(days=7)
dt.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0))

# if you need to round time
dt.datetime.now().replace(microsecond=0)

# convert second to datetime
dt.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

start_time = start_time_learning_r.replace('+00:00', '')   # start_time is string

# change to utc time
current_time = dt.datetime.now().replace(microsecond=0) + dt.timedelta(hours=hourdiff_to_utc)
import pytz;  xtime.astimezone(pytz.timezone('UTC'))    # this is for datetime 
df['time_col'].tz_convert(None)     # this is for pandas, tz_convert('UTC') for utc

dtime = (time1-time2).total_seconds()/86400.0   #86400=1day, time1 is datetime format 

# convert unix time to actual time
dt.datetime.utcfromtimestamp(1605761100).strftime('%Y-%m-%d %H:%M:%S')

# convert pandas column to datetime format
table['col'] = pd.to_datetime(table.datetime_col )   # include "utc=True" option to set it UTC
df['Time'] = df['Time'].astype('datetime64[ns]')     # If you see errors "Cannot compare tz-naive and tz-aware datetime-like objects" 
df['Time'] = df['Time'].dt.tz_convert(None)   # 
df['utc_datetime'].dt.date   # convert to date 

# calculate time of execution
import time
t1 = time.time(); t2 = time.time();
t2-t1

# convert from date to datetime
dt.datetime.combine( date, dt.datetime.min.time() )

# convert datetime to string 
x.strftime("%Y-%m-%d")
x.strftime("%Y-%m-%d %H:%M:%S")

# get specific weekday (Wednesday=2, Monday=0)
day = dt.date( 2020, 2, 3 )
wednesday = day + dt.timedelta( days=2+day.weekday() )  

# To convert from numpy.datetime64 to datetime
pd.to_datetime(x).to_pydatetime()
pd.to_datetime(x).to_pydatetime(tzinfo=None)  # remove timezone.

# convert from UTC to NewYork time.
x  = dt.datetime( 2022, 11, 1 ) 
local_tz = pytz.timezone('America/New_York')
x = x.astimezone( pytz.timezone('UTC') ).astimezone( local_tz ) 

# check if DayLight saving time or not.
x  = dt.datetime( 2022, 11, 1 ) 
local_tz = pytz.timezone('America/New_York')
x = x.astimezone( pytz.timezone('UTC') ).astimezone( local_tz ) 
dst_or_not = x.timetuple().tm_isdst

# get utc time now.
dt.utcnow()

#-------------------------------------------------------------


#---------- Pandas --------------------------------------------
# see pandas_summary.py
#--------------------------------------------------------------


#--------- numpy ----------------------------------------------
x = np.arange(2,10,2)
np.random.rand()

# drop dimension of 1
np.squeeze( M )

# expand dimenstion (put -1 to np.reshape)
np.reshape( -1, D )
np.expand_dims( x1, 0 )
np.stack( [img]*4, axis=2 )  # stack matrix by increasing dimension.
np.newaxis  

# shift index in array
t = np.roll( x123, 1,  axis=3 )  
#---------------------------------------------------------------

#--------- class ------------------------------------------------------
class MyClass():   # name cls/self can be anything.  cls/self is just convention
    class_attribute = "this is class attribute"
    def __init__(self, x):
        self.instance_attribute = x
    def get_instance_attribute(self):
        self.new_var = 'new variable'  # you can define new variable here too
        return self.instance_attribute
    @classmethod
    def get_class_attribute(cls):    
        return cls.class_attribute
    @classmethod
    def get_instance_attribute_on_class(cls):
        return cls.instance_attribute
    @staticmethod
    def get_static_attribute(y):
        return y

x = MyClass('this is instance attribute')
print( x.get_class_attribute() )
print( x.get_instance_attribute() )
print( x.get_static_attribute('this is static attribute') )
print( x.new_var )
#----------------------------------------------------------------------

#----- loading function from file --------------
from .xxx.file1 import class_name1   #load file.py
from .xxx.file2 import class_name2
#---- file1.py -----
class class_name1:
    @staticmethod
    def func1(xxx):
#---- executing -------
class_name1.func1(xxx)
#---------------------
#----- file2.py ----
class class_name2():
    def __init__(self, xxx): 
#-------------------
#---- executing ------
class2 = class_name2(xxx)
class2.func2()
#--------------------
#---------------------------------------------------
    
    
#--------- break, continue statement -------------------
for n in range(10):
    if n==5:
        continue  # continue will skip to next iteration loop
    else:
        break    # break will exit the for loop
#-------------------------------------------------------

#----- "try, exception" error usage ---------------
# https://camp.trainocate.co.jp/magazine/python-try-except/
# You could also use context manager instead of finally block.
    try:
        print(  int("fdsaf")   )
    except Exception as e:
        msg="=== Error @ agent_stats() initialization."
        raise Exception(msg,e)
    
    # do this if you want to bypass error.
    try:
        print( int("fdsaf") )
    except ValueError as e:
        pass  # used this if you want to ignore the error
    except xxxError as e:
        yyy
    except yyyError as e:
        zzz
    except:   # "except Exception as e:"   also works.
        print( int("39") )  # or do this. 
    else:   # In case the above error did not occur, this line is executed. 
        pass
    finally:  # This line executed no matter what.
        pass
        
raise Exception("exiting...")   # display error message and exit.  
#--------------------------------------------------

#------ env.yml -------------------------------------
API:
    url: address
    dir:
        xxx: 3
#---- run this env.yml ------------------------------
from envyaml import EnvYAML
vars = EnvYAML(...\env.yml)
var1 = vars.get("API.url")
var2 = vars.get("API.dir.xxx")
#----------------------------------------------------

#-------- delete all variable -----------------------
# https://www.kite.com/python/answers/how-to-delete-variables-and-functions-from-memory-in-python
def clear_var():
    for element in dir():
        if element[0:2] != "__":
        del globals()[element]
#----------------------------------------------------

#--------- save gif image ------------------------------------
clip = ImageSequenceClip(list(frames), fps=10)
clip.write_gif('test.gif', )
#-------------------------------------------------------------

#-------- use sys to get current dir ----------------
import sys
current_dir = sys.argv[1]
#----------------------------------------------------

#--------- function annotation ------------------------
def func(x: float) -> int:   # float, str, int, bool, 'pd.DataFrame', x (x is variable already defined)
    return int(x)
# basically informs readers func() returns integer
# x:float also informs readers x should be float.  That's all 
#-------------------------------------------------------

#---------- super class ---------------------------
class A:
	def __init__(self,name):
		self.a = "a"
        self.name = name
        
class main(A):  # dont forget the class name you want to load.
    def __init__(self):
        super().__init__(name)
        
xclass = main("xxx") 
print( xclass.a )
print( xclass.name ) 
#---------------------------------------------------

#-------------- class argument ---------------------
class A:
    def __init__(self, name):
        self.name = name
        
class main(A):
    def my_name(self):
        print("my name is ", self.name ) 

yclass = main("yyy")
print( yclass.my_name ) 
#--------------------------------------------------

#------ *args **kargs --------------------------
def fun( x, *xvars, **xdic ):
    print(x )
    print( xvars ) 
    print( xdic ) 
    return None

x = [ 1, 2, 3 ]
dic = { 'x':2, 'var':'yes' } 

fun( x, 1, 'hello', 40, text='yes' ) 
[1, 2, 3]
(1, 'hello', 40)
{'text': 'yes'}

fun( **dic ) 
2
()
{'var': 'yes'}
#------------------------------------------------

#------------ python class __add__ usage ---------------
class Dog:
    def __add__(self, x):
        return f'+ {x} called'
    def __sub__(self, x):
        return f'- {x} called'
    def __cal__(self,):
        # do something.

 dog = Dog()
 print( dog + 2 )  # + 2 called
 print( dog - 2 )  # - 2 called
#--------------------------------------------------------

#--------- dataclass ------------------------------------
from dataclasses import dataclass

@dataclass
class my_class:
  x:float = 2
  def fun(self):
    self.y = 3
    return self.x*2
#---------------------------------------------------------

#------- use of match case -------------------------------
def fun(status:int) -> str:
    match status:
        case 400:
	    return 1
        case 500:
	    return 2
        case _:
	    return 0
#---------------------------------------------------------

#---------- default dictionary --------------------------------
dic = {}
dic.setdefaults( "key_name", [] ).append(3)  # you can append values.
# Normally, dic2["key_name"] will show error if key_name doesn't exist, but
dic2 = defaultdict(int)  # you will avoid error.  
#-------------------------------------------------------------

#----------- Use dictionary instead of "if else" --------------------------
def func1(x,y):
    return x+y
def func2(x,y):
    return x-y

actions = {
    "start":func1,
    "stop":"func2,
    }

arg = (2, 3)
args = {"x":2, "y":3}
actions.get("start", "no outputs!!")(*arg, **args)
#----------------------------------------------------------------------------

#---------- tqdm (show progress bar) ---------------------------------
from tqdm import tqdm
with tqdm(enumerate(cols), total=len(cols)) as progress_bar:
    for n, col in progress_bar:
	...
#----------------------------------------------------------------------

#------------- context manager (decorator) ------------------------------
from contextlib import contextmanager
import time

@contextmanager
def performance_monitor():
    start = time.perf_counter()
    try:
        yield  # This is where your code will run
    finally:
        elapsed = time.perf_counter() - start
        print(f"Operation took {elapsed:.2f} seconds")
# Using our custom context manager
with performance_monitor():
    # Your complex operation here
    result = sum(i * i for i in range(1000000))
#-------------------------------------------------------------------------

#--------- Pipeline function ---------------------------------------------
class Pipeline:
    def __init__(self, *functions):
        self.functions = functions
        
    def __call__(self, x):
        for f in self.functions:
            x = f(x)
        return x

@Pipeline(
    lambda x: x.strip(),
    lambda x: x.replace('-', ''),
    lambda x: x.upper()
)
def clean_product_code(code: str) -> str:
    return code
# Usage
result = clean_product_code("  abc-123-xyz  ")  # Returns "ABC123XYZ"
#-----------------------------------------------------------------------

#------------ Partial usage --------------------------------------------
from functools import partial

# Define a regular function
def func(arg1, arg2, arg3):
    # Do something
    pass

# Create a partial function by fixing some arguments
partial_func = partial(func, fixed_arg1, fixed_arg2)

# Now call it with the remaining arguments
result = partial_func(arg3)
#------------------------------------------------------------------------
