import sqlite3

class Dbmanager(object):
	def __init__(self):
		self.conn = sqlite3.connect('ml_result.db')
		self.c = self.conn.cursor()

	def initdb(self):
		self.c.execute('''CREATE TABLE executions(exec_id integer, name text, accuracy real, exec_time real, Primary Key(exec_id))''')
		self.conn.commit()

	def new_row(self, name, accuracy, exec_time):
		self.c.execute('INSERT INTO executions(rowid, name, accuracy, exec_time) VALUES(NULL, " '+ name + '", "'+ str(accuracy) +'", "' + str(exec_time) +'")')
		self.conn.commit()
