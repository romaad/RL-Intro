class A(object):
    def foo(self, call_from: str):
        print("foo from A, call from %s" % call_from)


class B(object):
    def foo(self, call_from: str):
        print("foo from B, call from %s" % call_from)


class C(object):
    def foo(self, call_from: str):
        print("foo from C, call from %s" % call_from)


class D(A, B, C):

    pass


d = D()
d.foo("main")
