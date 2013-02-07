[TOC]

# 引言

本教程仅仅需要读者对Python有中等程度的掌握, 并不要求对程序的并行性有任何了解. 教程的目标是让你可以开始使用gevent, 帮助你解决现有的并发性问题并使你今天就能开始编写异步执行的程序.

### 作者

以时间顺序排列的作者：
[Stephen Diehl](http://www.stephendiehl.com)
[J&eacute;r&eacute;my Bethmont](https://github.com/jerem)
[sww](https://github.com/sww)
[Bruno Bigras](https://github.com/brunoqc)
[David Ripton](https://github.com/dripton)
[Travis Cline](https://github.com/traviscline)
[Boris Feld](https://github.com/Lothiraldan)
[youngsterxyf](https://github.com/youngsterxyf)
[Eddie Hebert](https://github.com/ehebert)
[Alexis Metaireau](http://notmyidea.org)
[Daniel Velkov](https://github.com/djv)

另外, 感谢gevent的作者Danis Bilenko给予本教程的支持.
Also thanks to Denis Bilenko for writing gevent and guidance in
constructing this tutorial.

本文是一个基于MIT许可证的合作文档.
要添加内容？发现了一个打印错误？请在[Github](https://github.com/sdiehl/gevent-tutorial)上fork并发起一个pull请求. 我们欢迎对本文的任何贡献.

### 中文译者
以时间顺序排列的翻译者：
[Han Liang](https://github.com/blurrcat)

中文翻译在[gevent-tutorial-ch](https://github.com/blurrcat/gevent-tutorial-ch), 欢迎指正翻译中的任何问题.

# 核心

## Greenlets

gevent使用的基本模式是<strong>Greenlet</strong>. Greenlet是一种轻量级的协同程序, 它作为一个C拓展模块被引入到Python之中. 所有的Greenlet都运行于一个系统进程之中, 它们之间的调度是合作的.

> 在任意时刻, 仅有一个greenlet在运行.

gevent的并行是与``multiprocessing``或``threading``等库提供的并行性是不同的. 那些库会通过操作系统来切换进程或POSIX线程, 以实现真正的并行.

## 同步与异步执行

并发性的核心想法是将一个任务分割成多个子任务, 并调度这些子任务同时或*异步*地执行, 而不是一次执行一个子任务, 即*同步*地执行. 这些子任务之间的切换被称为*上下文切换*.

在gevent中, 我们通过*让渡(yield)*来实现上下文切换. 在下面的例子中有两个上下文, 它们通过调用``gevent.sleep(0)``来相互切换.
A context switch in gevent is done through *yielding*. In this case
example we have two contexts which yield to each other through invoking
``gevent.sleep(0)``.

[[[cog
import gevent

def foo():
    print('Running in foo')
    gevent.sleep(0)
    print('Explicit context switch to foo again')

def bar():
    print('Explicit context to bar')
    gevent.sleep(0)
    print('Implicit context switch back to bar')

gevent.joinall([
    gevent.spawn(foo),
    gevent.spawn(bar),
])
]]]
[[[end]]]

通过下图可以清楚的看到上下文切换的发生, 你也可以使用一个调试器来跟踪程序的控制流.

![Greenlet Control Flow](flow.gif)

event的真正力量可以在网络或IO紧张的程序中得到最大发挥, 这类程序可以被合作的调度. Gevent会确保你的网络库在任何可能的时候隐式的让渡上下文. 我不能强调这有多么有用, 但下面的例子可以说明.

在这个例子中, ```select()``函数通常是一个阻塞的函数, 它对各种文件描述符进行轮询.

[[[cog
import time
import gevent
from gevent import select

start = time.time()
tic = lambda: 'at %1.1f seconds' % (time.time() - start)

def gr1():
    # 忙等待1秒, 但我们并不希望在这里阻塞..
    print('Started Polling: ', tic())
    select.select([], [], [], 2)
    print('Ended Polling: ', tic())

def gr2():
    # 忙等待1秒, 但我们并不希望在这里阻塞..
    print('Started Polling: ', tic())
    select.select([], [], [], 2)
    print('Ended Polling: ', tic())

def gr3():
    print("Hey lets do some stuff while the greenlets poll, at", tic())
    gevent.sleep(1)

gevent.joinall([
    gevent.spawn(gr1),
    gevent.spawn(gr2),
    gevent.spawn(gr3),
])
]]]
[[[end]]]

另一个例子定义了一个*非确定性*的``任务``函数(非确定性是指给定相同的输入, 函数不一定会给出相同的输出). 在这个例子中, 运行该任务函数会导致它暂停一段随机的时间.

[[[cog
import gevent
import random

def task(pid):
    """
    一个非确定性的任务
    """
    gevent.sleep(random.randint(0,2)*0.001)
    print('Task', pid, 'done')

def synchronous():
    for i in range(1,10):
        task(i)

def asynchronous():
    threads = [gevent.spawn(task, i) for i in xrange(10)]
    gevent.joinall(threads)

print('Synchronous:')
synchronous()

print('Asynchronous:')
asynchronous()
]]]
[[[end]]]

在同步函数``synchronous()``中, 所有的任务顺序执行, 每个任务执行时都会导致主程序*阻塞*(即暂停主程序的执行).

异步函数``asynchronous()``的重要部分是``gevent.spawn``, 它将一个给定的函数封装进一个greenlet线程中. 初始化的一列greenlet被存储于``threads``之中, 并被传递给``gevent.joinall``, 它将阻塞主程序的执行，直到所有的greenlet的终止.

一个重要的事实是，异步函数的执行顺序的随机的，并且其执行时间要远远少于同步函数. 实际上，同步函数的最大执行时间可能达到20秒，此时每个任务都暂停2秒；而异步函数的最大执行时间大致是2秒，因为没有哪个任务会阻塞其他任务的执行。

下面是一个更常见的例子，我们异步地从服务器获取数据，``fetch()``函数的执行时间在各个访问中可能不同，它与服务器在被访问时的负载有关。

<pre><code class="python">import gevent.monkey
gevent.monkey.patch_socket()

import gevent
import urllib2
import simplejson as json

def fetch(pid):
    response = urllib2.urlopen('http://json-time.appspot.com/time.json')
    result = response.read()
    json_result = json.loads(result)
    datetime = json_result['datetime']

    print 'Process ', pid, datetime
    return json_result['datetime']

def synchronous():
    for i in range(1,10):
        fetch(i)

def asynchronous():
    threads = []
    for i in range(1,10):
        threads.append(gevent.spawn(fetch, i))
    gevent.joinall(threads)

print 'Synchronous:'
synchronous()

print 'Asynchronous:'
asynchronous()
</code>
</pre>

## 确定性

在上文中我们提到，greenlet是具有确定性的。给定相同的配置的greenlet以及相同的输入，它们总是会给出相同的输出。在下面的例子中，我们将把一个任务分布到一个进程池中，并把它的结果与分布到gevent池(gevent pool)中的任务结果相比较。

<pre>
<code class="python">
import time

def echo(i):
    time.sleep(0.001)
    return i

# 非确定性的进程池

from multiprocessing.pool import Pool

p = Pool(10)
run1 = [a for a in p.imap_unordered(echo, xrange(10))]
run2 = [a for a in p.imap_unordered(echo, xrange(10))]
run3 = [a for a in p.imap_unordered(echo, xrange(10))]
run4 = [a for a in p.imap_unordered(echo, xrange(10))]

print( run1 == run2 == run3 == run4 )

# 决定性的gevent池

from gevent.pool import Pool

p = Pool(10)
run1 = [a for a in p.imap_unordered(echo, xrange(10))]
run2 = [a for a in p.imap_unordered(echo, xrange(10))]
run3 = [a for a in p.imap_unordered(echo, xrange(10))]
run4 = [a for a in p.imap_unordered(echo, xrange(10))]

print( run1 == run2 == run3 == run4 )
</code>
</pre>

<pre>
<code class="python">False
True</code>
</pre>

尽管gevent通常是确定性的，但当你开始与套接字或文件等外部服务进行交互时，不确定性因素可能进入你的程序。因此，即使绿色线程(green thread)是一种"决定性的并发", 你仍然可能会遇到一些在POSIX线程和进程中遇到的并发问题。

并发程序一个常见问题是*竞争条件(race condition)*. 简单的说，当两个并发的线程或进程依赖于某些共享资源并企图修改它时就会发生竞争条件。这将导致该共享资源的值与程序的执行顺序在时间上相关。这是一个应该尽量避免的问题，因为它使得程序的行为在全局上变得不确定。

解决这个问题的最好方式是在任何时候都避免使用全局状态。全局状态和导入时的副作用总会在某个时候回来咬你一口！

## 创建greenlet

gevent对greenlet的初始化提供了一些封装。下面是常见的使用模式:

[[[cog
import gevent
from gevent import Greenlet

def foo(message, n):
    """
    每个线程在初始化时会得到一个消息message，以及参数n.
    """
    gevent.sleep(n)
    print(message)

# 初始化一个greenlet实例来运行命名函数
# foo
thread1 = Greenlet.spawn(foo, "Hello", 1)

# 通过命名函数foo以及函数参数来创建和运行新的greenlet
thread2 = gevent.spawn(foo, "I live!", 2)

# Lambda表达式
thread3 = gevent.spawn(lambda x: (x+1), 2)

threads = [thread1, thread2, thread3]

# 阻塞主程序直到所有greenlet运行结束
gevent.joinall(threads)
]]]
[[[end]]]

除了使用基类Greenlet, 你也可以继承Greenlet类并覆盖 ``_run`` 方法。

[[[cog
import gevent
from gevent import Greenlet

class MyGreenlet(Greenlet):

    def __init__(self, message, n):
        Greenlet.__init__(self)
        self.message = message
        self.n = n

    def _run(self):
        print(self.message)
        gevent.sleep(self.n)

g = MyGreenlet("Hi there!", 3)
g.start()
g.join()
]]]
[[[end]]]


## Greenlet的状态

与其他任何代码相同，greenlet可能会在很多地方出错。一个greenlet可能抛出异常，停止，或消耗太多的系统资源。

greenlet的内部状态通常是时变的。下面是一些让你监督线程状态的greenlet标志:

- ``started`` -- Boolean, 表征Greenlet是否已经开始
- ``ready()`` -- Boolean, 表征Greenlet是否已经停止
- ``successful()`` -- Boolean, 表征Greenlet是否已经停止并且没有抛出任何异常
- ``value`` -- 任意值, Greenlet的返回值
- ``exception`` -- exception, greenlet内部抛出的、未捕捉的异常

[[[cog
import gevent

def win():
    return 'You win!'

def fail():
    raise Exception('You fail at failing.')

winner = gevent.spawn(win)
loser = gevent.spawn(fail)

print(winner.started) # True
print(loser.started)  # True

# Greenlet内部产生的异常会停留在greenlet之内
try:
    gevent.joinall([winner, loser])
except Exception as e:
    print('This will never be reached')

print(winner.value) # 'You win!'
print(loser.value)  # None

print(winner.ready()) # True
print(loser.ready())  # True

print(winner.successful()) # True
print(loser.successful())  # False

# fail函数抛出的异常不会传播到到greenlet之外。
# 一个堆栈跟踪会在标准输出上打印，但它不会延伸到父进程的堆栈。

print(loser.exception)

# 我们可以通过以下方式将异常再次抛出到greenlet之外:
# raise loser.exception
# 或
# loser.get()
]]]
[[[end]]]

## 程序的停止

在主程序收到SIGQUIT信号时，未能成功让渡的greenlet可能会使程序的执行比预期的更长。这将导致所谓的"僵尸进程"，它们必须在Python解释器之外终止。

这个问题的常见解决方式是在主程序中侦听SIGQUIT事件，并在退出之前调用``gevent.shutdown``.

<pre>
<code class="python">import gevent
import signal

def run_forever():
    gevent.sleep(1000)

if __name__ == '__main__':
    gevent.signal(signal.SIGQUIT, gevent.shutdown)
    thread = gevent.spawn(run_forever)
    thread.join()
</code>
</pre>

## 超时

超时是指对一段代码或一个greenlet在执行时间上的限制。

<pre>
<code class="python">
import gevent
from gevent import Timeout

seconds = 10

timeout = Timeout(seconds)
timeout.start()

def wait():
    gevent.sleep(10)

try:
    gevent.spawn(wait).join()
except Timeout:
    print 'Could not complete'

</code>
</pre>

超时也可以通过with表达式使用上下文管理器实现。

<pre>
<code class="python">import gevent
from gevent import Timeout

time_to_wait = 5 # seconds

class TooLong(Exception):
    pass

with Timeout(time_to_wait, TooLong):
    gevent.sleep(10)
</code>
</pre>

另外，gevent还为greenlet和许多数据结构提供了超时参数。例如:

[[[cog
import gevent
from gevent import Timeout

def wait():
    gevent.sleep(2)

timer = Timeout(1).start()
thread1 = gevent.spawn(wait)

try:
    thread1.join(timeout=timer)
except Timeout:
    print('Thread 1 timed out')

# --

timer = Timeout.start_new(1)
thread2 = gevent.spawn(wait)

try:
    thread2.get(timeout=timer)
except Timeout:
    print('Thread 2 timed out')

# --

try:
    gevent.with_timeout(1, wait)
except Timeout:
    print('Thread 3 timed out')

]]]
[[[end]]]

## 猴子补丁(Monkeypatching)

唉，我们现在到了Gevent的黑暗角落。到现在为止，我一直试图激发你使用强大的协同程序模式，并避免提到猴子补丁。但现在我们不得不对猴子补丁的黑暗艺术进行讨论了。可能你已经注意到我们在上面的例子中调用了``monkey.patch_socket()``这个命令。这是一个完全只有副作用的命令，它将修改标准库中的套接字库。

<pre>
<code class="python">import socket
print( socket.socket )

print "After monkey patch"
from gevent import monkey
monkey.patch_socket()
print( socket.socket )

import select
print select.select
monkey.patch_select()
print "After monkey patch"
print( select.select )
</code>
</pre>

<pre>
<code class="python">class 'socket.socket'
After monkey patch
class 'gevent.socket.socket'

built-in function select
After monkey patch
function select at 0x1924de8
</code>
</pre>

Python允许大多数对象在运行时被修改，包括模块，类，甚至是函数。通常来说，这是个令人震惊的坏主意，因为它会产生"隐式副作用"。在问题出现时它将使得debug变得极其困难。然而，在极端情况下，一个库可能需要修改Python本身的行为，这时就可以使用猴子补丁。gevent对大多数阻塞的系统调用都打了补丁，包括``socket``, ``ssl``, ``threading`` 和 ``select``中的各个函数。这使得这些库的行为变得合作。

例如，Redis的python绑定通常使用普通的tcp套接字来与``redis-server``进行通信。通过简单的调用``gevent.monkey.patch_all()``，我们就可以让redis合作的调度各个访问，并与gevent的其他部分正常的工作。

这使得许多通常不能与gevent一起工作的库可以与gevent进行整合，而我们甚至一行代码都不用写。尽管猴子补丁是罪恶的，但在这里它仍然是"有用的罪恶"。

# 数据结构

## 事件(Events)

事件是一种greenlet之间进行异步通讯的方式。

<pre>
<code class="python">import gevent
from gevent.event import AsyncResult

a = AsyncResult()

def setter():
    """
    在3秒后对所有等待a的值的线程进行set
    """
    gevent.sleep(3)
    a.set()

def waiter():
    """
    在3秒后get调用将停止阻塞
    """
    a.get() # 阻塞
    print 'I live!'

gevent.joinall([
    gevent.spawn(setter),
    gevent.spawn(waiter),
])

</code>
</pre>

对Event对象的一个扩展是AsyncResult(异步结果)，它允许你发送一个值以及一个唤醒调用。它有时又被称作未来(future)或者延迟(deferred), 因为它持有一个对未来值的引用，并可以在安排在任意时刻对这个值进行修改。

<pre>
<code class="python">import gevent
from gevent.event import AsyncResult
a = AsyncResult()

def setter():
    """
    在3秒后设置a的值
    """
    gevent.sleep(3)
    a.set('Hello!')

def waiter():
    """
    在3秒后，a的值会被设置，此后get调用将停止阻塞
    """
    print a.get()

gevent.joinall([
    gevent.spawn(setter),
    gevent.spawn(waiter),
])

</code>
</pre>

## 队列(Queues)

Queues are ordered sets of data that have the usual ``put`` / ``get``
operations but are written in a way such that they can be safely
manipulated across Greenlets.

For example if one Greenlet grabs an item off of the queue, the
same item will not grabbed by another Greenlet executing
simultaneously.

[[[cog
import gevent
from gevent.queue import Queue

tasks = Queue()

def worker(n):
    while not tasks.empty():
        task = tasks.get()
        print('Worker %s got task %s' % (n, task))
        gevent.sleep(0)

    print('Quitting time!')

def boss():
    for i in xrange(1,25):
        tasks.put_nowait(i)

gevent.spawn(boss).join()

gevent.joinall([
    gevent.spawn(worker, 'steve'),
    gevent.spawn(worker, 'john'),
    gevent.spawn(worker, 'nancy'),
])
]]]
[[[end]]]

Queues can also block on either ``put`` or ``get`` as the need arises. 

Each of the ``put`` and ``get`` operations has a non-blocking
counterpart, ``put_nowait`` and 
``get_nowait`` which will not block, but instead raise
either ``gevent.queue.Empty`` or
``gevent.queue.Full`` in the operation is not possible.

In this example we have the boss running simultaneously to the
workers and have a restriction on the Queue preventing it from containing
more than three elements. This restriction means that the ``put``
operation will block until there is space on the queue.
Conversely the ``get`` operation will block if there are
no elements on the queue to fetch, it also takes a timeout
argument to allow for the queue to exit with the exception
``gevent.queue.Empty`` if no work can found within the
time frame of the Timeout.

[[[cog
import gevent
from gevent.queue import Queue, Empty

tasks = Queue(maxsize=3)

def worker(n):
    try:
        while True:
            task = tasks.get(timeout=1) # decrements queue size by 1
            print('Worker %s got task %s' % (n, task))
            gevent.sleep(0)
    except Empty:
        print('Quitting time!')

def boss():
    """
    Boss will wait to hand out work until a individual worker is
    free since the maxsize of the task queue is 3.
    """

    for i in xrange(1,10):
        tasks.put(i)
    print('Assigned all work in iteration 1')

    for i in xrange(10,20):
        tasks.put(i)
    print('Assigned all work in iteration 2')

gevent.joinall([
    gevent.spawn(boss),
    gevent.spawn(worker, 'steve'),
    gevent.spawn(worker, 'john'),
    gevent.spawn(worker, 'bob'),
])
]]]
[[[end]]]

## Groups and Pools

A group is a collection of running greenlets which are managed
and scheduled together as group. It also doubles as parallel
dispatcher that mirrors the Python ``multiprocessing`` library.

[[[cog
import gevent
from gevent.pool import Group

def talk(msg):
    for i in xrange(3):
        print(msg)

g1 = gevent.spawn(talk, 'bar')
g2 = gevent.spawn(talk, 'foo')
g3 = gevent.spawn(talk, 'fizz')

group = Group()
group.add(g1)
group.add(g2)
group.join()

group.add(g3)
group.join()
]]]
[[[end]]]

This is very useful for managing groups of asynchronous tasks.

As mentioned above, ``Group`` also provides an API for dispatching
jobs to grouped greenlets and collecting their results in various
ways.

[[[cog
import gevent
from gevent import getcurrent
from gevent.pool import Group

group = Group()

def hello_from(n):
    print('Size of group', len(group))
    print('Hello from Greenlet %s' % id(getcurrent()))

group.map(hello_from, xrange(3))


def intensive(n):
    gevent.sleep(3 - n)
    return 'task', n

print('Ordered')

ogroup = Group()
for i in ogroup.imap(intensive, xrange(3)):
    print(i)

print('Unordered')

igroup = Group()
for i in igroup.imap_unordered(intensive, xrange(3)):
    print(i)

]]]
[[[end]]]

A pool is a structure designed for handling dynamic numbers of
greenlets which need to be concurrency-limited.  This is often
desirable in cases where one wants to do many network or IO bound
tasks in parallel.

[[[cog
import gevent
from gevent.pool import Pool

pool = Pool(2)

def hello_from(n):
    print('Size of pool', len(pool))

pool.map(hello_from, xrange(3))
]]]
[[[end]]]

Often when building gevent driven services one will center the
entire service around a pool structure. An example might be a
class which polls on various sockets.

<pre>
<code class="python">from gevent.pool import Pool

class SocketPool(object):

    def __init__(self):
        self.pool = Pool(1000)
        self.pool.start()

    def listen(self, socket):
        while True:
            socket.recv()

    def add_handler(self, socket):
        if self.pool.full():
            raise Exception("At maximum pool size")
        else:
            self.pool.spawn(self.listen, socket)

    def shutdown(self):
        self.pool.kill()

</code>
</pre>

## Locks and Semaphores

A semaphore is a low level synchronization primitive that allows
greenlets to coordinate and limit concurrent access or execution. A
semaphore exposes two methods, ``acquire`` and ``release`` The
difference between the number of times and a semaphore has been
acquired and released is called the bound of the semaphore. If a
semaphore bound reaches 0 it will block until another greenlet
releases its acquisition.

[[[cog
from gevent import sleep
from gevent.pool import Pool
from gevent.coros import BoundedSemaphore

sem = BoundedSemaphore(2)

def worker1(n):
    sem.acquire()
    print('Worker %i acquired semaphore' % n)
    sleep(0)
    sem.release()
    print('Worker %i released semaphore' % n)

def worker2(n):
    with sem:
        print('Worker %i acquired semaphore' % n)
        sleep(0)
    print('Worker %i released semaphore' % n)

pool = Pool()
pool.map(worker1, xrange(0,2))
pool.map(worker2, xrange(3,6))
]]]
[[[end]]]

A semaphore with bound of 1 is known as a Lock. it provides
exclusive execution to one greenlet. They are often used to
ensure that resources are only in use at one time in the context
of a program.

## Thread Locals

Gevent also allows you to specify data which is local to the
greenlet context. Internally, this is implemented as a global
lookup which addresses a private namespace keyed by the
greenlet's ``getcurrent()`` value.

[[[cog
import gevent
from gevent.local import local

stash = local()

def f1():
    stash.x = 1
    print(stash.x)

def f2():
    stash.y = 2
    print(stash.y)

    try:
        stash.x
    except AttributeError:
        print("x is not local to f2")

g1 = gevent.spawn(f1)
g2 = gevent.spawn(f2)

gevent.joinall([g1, g2])
]]]
[[[end]]]

Many web framework thats integrate with gevent store HTTP session
objects inside of gevent thread locals. For example using the
Werkzeug utility library and its proxy object we can create
Flask style request objects.

<pre>
<code class="python">from gevent.local import local
from werkzeug.local import LocalProxy
from werkzeug.wrappers import Request
from contextlib import contextmanager

from gevent.wsgi import WSGIServer

_requests = local()
request = LocalProxy(lambda: _requests.request)

@contextmanager
def sessionmanager(environ):
    _requests.request = Request(environ)
    yield
    _requests.request = None

def logic():
    return "Hello " + request.remote_addr

def application(environ, start_response):
    status = '200 OK'

    with sessionmanager(environ):
        body = logic()

    headers = [
        ('Content-Type', 'text/html')
    ]

    start_response(status, headers)
    return [body]

WSGIServer(('', 8000), application).serve_forever()


<code>
</pre>

Flask's system is a bit more sophisticated than this example, but the
idea of using thread locals as local session storage is nonetheless the
same.

## Subprocess

As of gevent 1.0, ``gevent.subprocess`` -- a patched version of Python's
``subprocess`` module -- has been added. It supports cooperative waiting on
subprocesses.

<pre>
<code class="python">
import gevent
from gevent.subprocess import Popen, PIPE

def cron():
    while True:
        print "cron"
        gevent.sleep(0.2)

g = gevent.spawn(cron)
sub = Popen(['sleep 1; uname'], stdout=PIPE, shell=True)
out, err = sub.communicate()
g.kill()
print out.rstrip()
</pre>

<pre>
<code class="python">
cron
cron
cron
cron
cron
Linux
<code>
</pre>

Many people also want to use ``gevent`` and ``multiprocessing`` together. One of
the most obvious challenges is that inter-process communication provided by
``multiprocessing`` is not cooperative by default. Since
``multiprocessing.Connection``-based objects (such as ``Pipe``) expose their
underlying file descriptors, ``gevent.socket.wait_read`` and ``wait_write`` can
be used to cooperatively wait for ready-to-read/ready-to-write events before
actually reading/writing:

<pre>
<code class="python">
import gevent
from multiprocessing import Process, Pipe
from gevent.socket import wait_read, wait_write

# To Process
a, b = Pipe()

# From Process
c, d = Pipe()

def relay():
    for i in xrange(10):
        msg = b.recv()
        c.send(msg + " in " + str(i))

def put_msg():
    for i in xrange(10):
        wait_write(a.fileno())
        a.send('hi')

def get_msg():
    for i in xrange(10):
        wait_read(d.fileno())
        print(d.recv())

if __name__ == '__main__':
    proc = Process(target=relay)
    proc.start()

    g1 = gevent.spawn(get_msg)
    g2 = gevent.spawn(put_msg)
    gevent.joinall([g1, g2], timeout=1)
</code>
</pre>

Note, however, that the combination of ``multiprocessing`` and gevent brings
along certain OS-dependent pitfalls, among others:

* After [forking](http://linux.die.net/man/2/fork) on POSIX-compliant systems
gevent's state in the child is ill-posed. One side effect is that greenlets
spawned before ``multiprocessing.Process`` creation run in both, parent and
child process.
* ``a.send()`` in ``put_msg()`` above might still block the calling thread
non-cooperatively: a ready-to-write event only ensures that one byte can be
written. The underlying buffer might be full before the attempted write is
complete.
* The ``wait_write()`` / ``wait_read()``-based approach as indicated above does
not work on Windows (``IOError: 3 is not a socket (files are not supported)``),
because Windows cannot watch pipes for events.

The Python package [gipc](http://pypi.python.org/pypi/gipc) overcomes these
challenges for you in a largely transparent fashion on both, POSIX-compliant and
Windows systems. It provides gevent-aware ``multiprocessing.Process``-based
child processes and gevent-cooperative inter-process communication based on
pipes.

## Actors

The actor model is a higher level concurrency model popularized
by the language Erlang. In short the main idea is that you have a
collection of independent Actors which have an inbox from which
they receive messages from other Actors. The main loop inside the
Actor iterates through its messages and takes action according to
its desired behavior. 

Gevent does not have a primitive Actor type, but we can define
one very simply using a Queue inside of a subclassed Greenlet.

<pre>
<code class="python">import gevent
from gevent.queue import Queue


class Actor(gevent.Greenlet):

    def __init__(self):
        self.inbox = Queue()
        Greenlet.__init__(self)

    def receive(self, message):
        """
        Define in your subclass.
        """
        raise NotImplemented()

    def _run(self):
        self.running = True

        while self.running:
            message = self.inbox.get()
            self.receive(message)

</code>
</pre>

In a use case:

<pre>
<code class="python">import gevent
from gevent.queue import Queue
from gevent import Greenlet

class Pinger(Actor):
    def receive(self, message):
        print message
        pong.inbox.put('ping')
        gevent.sleep(0)

class Ponger(Actor):
    def receive(self, message):
        print message
        ping.inbox.put('pong')
        gevent.sleep(0)

ping = Pinger()
pong = Ponger()

ping.start()
pong.start()

ping.inbox.put('start')
gevent.joinall([ping, pong])
</code>
</pre>

# Real World Applications

## Gevent ZeroMQ

[ZeroMQ](http://www.zeromq.org/) is described by its authors as
"a socket library that acts as a concurrency framework". It is a
very powerful messaging layer for building concurrent and
distributed applications. 

ZeroMQ provides a variety of socket primitives, the simplest of
which being a Request-Response socket pair. A socket has two
methods of interest ``send`` and ``recv``, both of which are
normally blocking operations. But this is remedied by a briliant
library by [Travis Cline](https://github.com/traviscline) which
uses gevent.socket to poll ZeroMQ sockets in a non-blocking
manner.  You can install gevent-zeromq from PyPi via:  ``pip install
gevent-zeromq``

[[[cog
# Note: Remember to ``pip install pyzmq gevent_zeromq``
import gevent
from gevent_zeromq import zmq

# Global Context
context = zmq.Context()

def server():
    server_socket = context.socket(zmq.REQ)
    server_socket.bind("tcp://127.0.0.1:5000")

    for request in range(1,10):
        server_socket.send("Hello")
        print('Switched to Server for ', request)
        # Implicit context switch occurs here
        server_socket.recv()

def client():
    client_socket = context.socket(zmq.REP)
    client_socket.connect("tcp://127.0.0.1:5000")

    for request in range(1,10):

        client_socket.recv()
        print('Switched to Client for ', request)
        # Implicit context switch occurs here
        client_socket.send("World")

publisher = gevent.spawn(server)
client    = gevent.spawn(client)

gevent.joinall([publisher, client])

]]]
[[[end]]]

## Simple Servers

<pre>
<code class="python">
# On Unix: Access with ``$ nc 127.0.0.1 5000`` 
# On Window: Access with ``$ telnet 127.0.0.1 5000`` 

from gevent.server import StreamServer

def handle(socket, address):
    socket.send("Hello from a telnet!\n")
    for i in range(5):
        socket.send(str(i) + '\n')
    socket.close()

server = StreamServer(('127.0.0.1', 5000), handle)
server.serve_forever()
</code>
</pre>

## WSGI Servers

Gevent provides two WSGI servers for serving content over HTTP.
Henceforth called ``wsgi`` and ``pywsgi``:

* gevent.wsgi.WSGIServer
* gevent.pywsgi.WSGIServer

In earlier versions of gevent before 1.0.x, gevent used libevent
instead of libev. Libevent included a fast HTTP server which was
used by gevent's ``wsgi`` server. 

In gevent 1.0.x there is no http server included. Instead
``gevent.wsgi`` is now an alias for the pure Python server in
``gevent.pywsgi``.


## Streaming Servers

**If you are using gevent 1.0.x, this section does not apply**

For those familiar with streaming HTTP services, the core idea is
that in the headers we do not specify a length of the content. We
instead hold the connection open and flush chunks down the pipe,
prefixing each with a hex digit indicating the length of the
chunk. The stream is closed when a size zero chunk is sent.

    HTTP/1.1 200 OK
    Content-Type: text/plain
    Transfer-Encoding: chunked

    8
    <p>Hello

    9
    World</p>

    0

The above HTTP connection could not be created in wsgi
because streaming is not supported. It would instead have to
buffered.

<pre>
<code class="python">from gevent.wsgi import WSGIServer

def application(environ, start_response):
    status = '200 OK'
    body = '&lt;p&gt;Hello World&lt;/p&gt;'

    headers = [
        ('Content-Type', 'text/html')
    ]

    start_response(status, headers)
    return [body]

WSGIServer(('', 8000), application).serve_forever()

</code>
</pre> 

Using pywsgi we can however write our handler as a generator and
yield the result chunk by chunk.

<pre>
<code class="python">from gevent.pywsgi import WSGIServer

def application(environ, start_response):
    status = '200 OK'

    headers = [
        ('Content-Type', 'text/html')
    ]

    start_response(status, headers)
    yield "&lt;p&gt;Hello"
    yield "World&lt;/p&gt;"

WSGIServer(('', 8000), application).serve_forever()

</code>
</pre> 

But regardless, performance on Gevent servers is phenomenal
compared to other Python servers. libev is a very vetted technology
and its derivative servers are known to perform well at scale.

To benchmark, try Apache Benchmark ``ab`` or see this 
[Benchmark of Python WSGI Servers](http://nichol.as/benchmark-of-python-web-servers) 
for comparison with other servers.

<pre>
<code class="shell">$ ab -n 10000 -c 100 http://127.0.0.1:8000/
</code>
</pre> 

## Long Polling

<pre>
<code class="python">import gevent
from gevent.queue import Queue, Empty
from gevent.pywsgi import WSGIServer
import simplejson as json

data_source = Queue()

def producer():
    while True:
        data_source.put_nowait('Hello World')
        gevent.sleep(1)

def ajax_endpoint(environ, start_response):
    status = '200 OK'
    headers = [
        ('Content-Type', 'application/json')
    ]

    start_response(status, headers)

    while True:
        try:
            datum = data_source.get(timeout=5)
            yield json.dumps(datum) + '\n'
        except Empty:
            pass


gevent.spawn(producer)

WSGIServer(('', 8000), ajax_endpoint).serve_forever()

</code>
</pre>

## Websockets

Websocket example which requires <a href="https://bitbucket.org/Jeffrey/gevent-websocket/src">gevent-websocket</a>.


<pre>
<code class="python"># Simple gevent-websocket server
import json
import random

from gevent import pywsgi, sleep
from geventwebsocket.handler import WebSocketHandler

class WebSocketApp(object):
    '''Send random data to the websocket'''

    def __call__(self, environ, start_response):
        ws = environ['wsgi.websocket']
        x = 0
        while True:
            data = json.dumps({'x': x, 'y': random.randint(1, 5)})
            ws.send(data)
            x += 1
            sleep(0.5)

server = pywsgi.WSGIServer(("", 10000), WebSocketApp(),
    handler_class=WebSocketHandler)
server.serve_forever()
</code>
</pre>

HTML Page:

    <html>
        <head>
            <title>Minimal websocket application</title>
            <script type="text/javascript" src="jquery.min.js"></script>
            <script type="text/javascript">
            $(function() {
                // Open up a connection to our server
                var ws = new WebSocket("ws://localhost:10000/");

                // What do we do when we get a message?
                ws.onmessage = function(evt) {
                    $("#placeholder").append('<p>' + evt.data + '</p>')
                }
                // Just update our conn_status field with the connection status
                ws.onopen = function(evt) {
                    $('#conn_status').html('<b>Connected</b>');
                }
                ws.onerror = function(evt) {
                    $('#conn_status').html('<b>Error</b>');
                }
                ws.onclose = function(evt) {
                    $('#conn_status').html('<b>Closed</b>');
                }
            });
        </script>
        </head>
        <body>
            <h1>WebSocket Example</h1>
            <div id="conn_status">Not Connected</div>
            <div id="placeholder" style="width:600px;height:300px;"></div>
        </body>
    </html>


## Chat Server

The final motivating example, a realtime chat room. This example
requires <a href="http://flask.pocoo.org/">Flask</a> ( but not neccesarily so, you could use Django,
Pyramid, etc ). The corresponding Javascript and HTML files can
be found <a href="https://github.com/sdiehl/minichat">here</a>.


<pre>
<code class="python"># Micro gevent chatroom.
# ----------------------

from flask import Flask, render_template, request

from gevent import queue
from gevent.pywsgi import WSGIServer

import simplejson as json

app = Flask(__name__)
app.debug = True

rooms = {
    'topic1': Room(),
    'topic2': Room(),
}

users = {}

class Room(object):

    def __init__(self):
        self.users = set()
        self.messages = []

    def backlog(self, size=25):
        return self.messages[-size:]

    def subscribe(self, user):
        self.users.add(user)

    def add(self, message):
        for user in self.users:
            print user
            user.queue.put_nowait(message)
        self.messages.append(message)

class User(object):

    def __init__(self):
        self.queue = queue.Queue()

@app.route('/')
def choose_name():
    return render_template('choose.html')

@app.route('/&lt;uid&gt;')
def main(uid):
    return render_template('main.html',
        uid=uid,
        rooms=rooms.keys()
    )

@app.route('/&lt;room&gt;/&lt;uid&gt;')
def join(room, uid):
    user = users.get(uid, None)

    if not user:
        users[uid] = user = User()

    active_room = rooms[room]
    active_room.subscribe(user)
    print 'subscribe', active_room, user

    messages = active_room.backlog()

    return render_template('room.html',
        room=room, uid=uid, messages=messages)

@app.route("/put/&lt;room&gt;/&lt;uid&gt;", methods=["POST"])
def put(room, uid):
    user = users[uid]
    room = rooms[room]

    message = request.form['message']
    room.add(':'.join([uid, message]))

    return ''

@app.route("/poll/&lt;uid&gt;", methods=["POST"])
def poll(uid):
    try:
        msg = users[uid].queue.get(timeout=10)
    except queue.Empty:
        msg = []
    return json.dumps(msg)

if __name__ == "__main__":
    http = WSGIServer(('', 5000), app)
    http.serve_forever()
</code>
</pre>
