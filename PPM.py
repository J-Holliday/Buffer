from PIL import Image
import math
import numpy as np

class PPM(object):
    """
    Sample:
    [making ppm simply]
    > ppm = PPM()
    > print(ppm.executor(ppm.new_image()))
    [buffering]
    > ppm = PPM()
    > while flag:
    >     if ppm.buffering((val0, val1)):
    >         ppm.executor(ppm.new_image())
    """

    def __init__(self):
        self.buf_idx = 0
        self.buf_error = [0 for i in range(16)]
        self.buf_grad = [0 for i in range(16)]
        self.means_idx = 0
        self.error_1000 = []
        self.grad_1000 = []
        self.error_means = []
        self.grad_means = []
    
    def new_image(self):
        im = Image.new("RGB", (32, 16))
        return im
    
    def buffering(self, values):
        """
        Type:
        values is tuple of (error_val, grad_val)
        And return is flag of executor.
        """

        self.buf_error[self.buf_idx] = self.encoder(values[0], True)
        self.buf_grad[self.buf_idx] = self.encoder(values[1], False)
        #print("error:{e}, grad:{g}".format(e=values[0], g=values[1]))
        
        self.buf_idx += 1
        if self.buf_idx == 16:
            self.buf_idx = 0
            return True
        else:
            return False
    
    def buffering2(self, values, n):
        """
        Type:
        values is tuple of (error_val, grad_val)
        And return is flag of executor.
        buffering2 differ in buffer algorithm from buffering.
        buffering2 hold values of means at recently 1000 times.
        """

        self.buf_error[self.buf_idx] = self.encoder(values[0], True)
        self.buf_grad[self.buf_idx] = self.encoder(values[1], False)
        #print("error:{e}, grad:{g}".format(e=values[0], g=values[1]))
        
        # calc meanvalue
        if len(self.error_1000) == 1000:
            self.error_means.append(np.array(self.error_1000).mean())
            self.means_idx += 1
            self.error_1000 = []
        self.error_1000.append(values[0])

        if len(self.grad_1000) == 1000:
            self.grad_means.append(np.array(self.grad_1000).mean())
            self.means_idx += 1
            self.grad_1000 = []
        self.grad_1000.append(values[1])

				# override meanvalue
        if len(self.error_means) > 15:
            for i in range(16):
                #print("overriding..."+str(i))
                self.buf_error[i] = self.encoder(self.error_means[-(16-i)], True)
                self.buf_grad[i] = self.encoder(self.grad_means[-(16-i)], False)
        else:
            for i in range(len(self.error_means)):
                self.buf_error[i] = self.encoder(self.error_means[i], True)
                self.buf_grad[i] = self.encoder(self.grad_means[i], False)

        self.buf_idx += 1
        if self.buf_idx == 16:
            self.buf_idx = 0
            return True
        else:
            return False
    
    def encoder(self, value, signed):
        signed_bar = [1/16. * i / 2. for i in range(16)]
        unsigned_bar = [(1/16. * i - 0.5) * 10 for i in range(16)]
        #u1 = [math.exp(i) for i in range(8)]
        #u2 = [-math.exp(i) for i in range(8)]
        #unsigned_bar = u1+u2
        #unsigned_bar.sort()

        if signed == True:
            for i in range(15):
                if value < 0.0:
                    return 15
                elif signed_bar[i] < value < signed_bar[i+1]:
                    return 15-i
            return 0
        else:
            for i in range(15):
                if value < unsigned_bar[0]:
                    return 15
                elif unsigned_bar[i] < value < unsigned_bar[i+1]:
                    return 15-i
            return 0
                
    def error(self, dummy=True):
        x = [i for i in range(0, 16)]
        if dummy:
            y = [i for i in range(15, -1, -1)]
        else:
            y = self.buf_error
        color = []
        for i in range(16):
            if y[i] > 10:
                color.append((0, 255, 0))
            else:
                color.append((0, 0, 255))
        return ((x, y), color)
    
    def grad(self, dummy=True):
        x = [i for i in range(16, 32)]
        if dummy:
            y = [i for i in range(0, 16)]
        else:
            y = self.buf_grad
        color = []
        for i in range(16):
            if y[i] <= 7:
                color.append((0, 255, 0))
            else:
                color.append((255, 0, 0))
        return ((x, y), color)
    
    def executor(self, im, filename="output"):
        m1, c1 = self.error(dummy=False)
        x1 = m1[0]; y1 = m1[1]
        m2, c2 = self.grad(dummy=False)
        x2 = m2[0]; y2 = m2[1]
        x = x1+x2; y = y1+y2; c = c1+c2
        for i in range(len(x)):
            im.putpixel((x[i], y[i]), c[i])
        try:
            im.save("{fn}.ppm".format(fn=filename))
            res = True
        except:
            print("Error in PPM.executor()")
            res = False
        finally:
            return res
