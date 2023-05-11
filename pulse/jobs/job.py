from .pendingio import PendingIO

class Job(PendingIO):
    REGISTER_KEYWORD = 'eval_ mode:'

if __name__ == '__main__':
    a = Job(name='eval_ mode: Etest,Vcpu name: test1')
    print(a.start)
    b = Job(name='eval_ mode: Etest,Vcpu name: test2')
    print(b.start)
    c = Job(name='eval_ mode: Etest,Vcpu,C name: test3')
    cfile = c.file
    print(c.start)
    print(c.end({"a": 2}))
    print(Job(file=c.file).load())