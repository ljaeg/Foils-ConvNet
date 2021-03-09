#This is a program for going thru the SAH website and collecting amazon codes of images of craters 
#To store on the machine at Berkeley.

# The first step is to just get a list of 

import mysql.connector
import os

mydb = mysql.connector.connect(
  host="flair.ssl.berkeley.edu",
  user="stardust",
  passwd="56y$Uq2CY",
  database="stardust"
)


Dir = os.path.join("..", "Data")
fname = "amazon20k.txt"

# tech = 1 means calibration crater.
# conf is the number of times someone has confirmed a crater is present.
# disconf is the number of times someone has confirmed there is no crater
# bad_focus is the number of times someone reported the focus was bad
# clickfraction = conf/(disconf+conf)
# Query to get non-calibration images without craters.
query = "SELECT amazon_key FROM `real_movie` WHERE tech = 0 and disconf > 5 AND conf < 3 AND bad_focus < 50 LIMIT 2000"
# Query to get calibration images with craters.
query = "SELECT amazon_key FROM `real_movie` WHERE tech = 1 and conf > 5 AND disconf < 3 AND bad_focus < 50 LIMIT 2000"
# or if this particular database has the clickfraction code:
query = "SELECT amazon_key FROM `real_movie` WHERE tech = 1 and clickfraction > 0.5 AND bad_focus < 50 LIMIT 2000"
# Query to get random blank images.
query = "SELECT amazon_key FROM `real_movie` WHERE exclude = '' AND comment = '' AND tech = 0 AND bad_focus < 50 LIMIT 20000"
file = open(Dir + fname + ".txt", "w")
cursor = mydb.cursor()
cursor.execute(query)
result = cursor.fetchall()
i = 0
for key in result:
	i += 1
	file.write(key[0])
	file.write("\n")
print(i)
file.close()

