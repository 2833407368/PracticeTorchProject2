class Person:
    def __call__(self,name):
        print("hi"+name)
    def hello(self,name):
        print("hello"+name)
person = Person()
person("zhangsan")
person.hello("zhangsan")