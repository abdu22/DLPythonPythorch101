class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def move(self):
        print(f'move {self.x} to {self.y}')

    def draw(self):
        print('draw')

point1 = Point(1,2)
point1.move()
print(point1.x)
