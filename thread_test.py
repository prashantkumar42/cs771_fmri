import _thread
import time


beta = 5


def fun(alpha):
    for i in range(alpha):
        print(str(alpha)+":"+str(beta))


alpha_lasso = [1, 5, 7, 9]
for alpha_cur in alpha_lasso:
    time.sleep(1)
    try:
        print("created thread for alpha: " + str(alpha_cur))
        _thread.start_new_thread(fun, (alpha_cur,))
    except:
        print("Error spawning new thread")
