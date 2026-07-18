#--------- tasks.py ------------------
from celery import Celery

app = Celery(
    "tasks",
    broker="redis://localhost:6379/0"
)

@app.task
def add(x, y):
    return x + y

#-------- Call it from application -------
result = add.delay(5, 7)
print(result.id)

#------ Get result later -----------------
print(result.get())
