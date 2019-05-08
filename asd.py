def a():
    return def b(): print("a")

c = a()
c()