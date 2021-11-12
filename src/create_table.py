import sqlite3

dbname = './register.db'
conn = sqlite3.connect(dbname)
cur = conn.cursor()

create_table = '''
create table IF NOT EXISTS register_people(
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT,
data TEXT
);
'''
insert_sql = "INSERT INTO register_people (name,data) values('test', 'test')"

select_sql = 'SELECT id,name FROM register_people'

cur.execute(create_table)
#cur.execute(insert_sql)
# cur.execute(select_sql)
# out = cur.fetchall()
# print(out)
conn.commit()

conn.close()
