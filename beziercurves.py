from scipy.special import binom
from PyQt4 import QtCore
from matplotlib.lines import Line2D
import numpy as np

class BezierBuilder(QtCore.QObject):
    """ CLASS - Interactive bezier curve (add/remove points(ctrl-click), drag individual points or entire curve)
        Update curve while the points are being dragged, and emit a signal when a drag/change is finished
        Based on and expanded from Source: https://gist.github.com/Juanlu001/7284462
    """
    changed = QtCore.pyqtSignal(bool)

    def __init__(self,control_polygon,ax,points,linedrag=False,fixpoints=[]):
        """
        Receives the initial control polygon of the curve, the figure axis, the number of curve points,
        Defines whether the bezier curve itself can be dragged, and which control points cannot be dragged.
        """
        QtCore.QObject.__init__(self)

        self.control_polygon = control_polygon
        self.xp = list(control_polygon.get_xdata())
        self.yp = list(control_polygon.get_ydata())
        self.canvas = control_polygon.figure.canvas
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.canvas.setFocus()

        self.axis = ax
        self.np = points
        self.lineDrag = linedrag
        self.fixPoints = fixpoints
        self.isRemoved = False

        # Event handler for mouse and keyboard interaction
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion_notify)

        # Initialize Bezier curve and add to the figure axis
        line_bezier = Line2D([], [], c=control_polygon.get_markeredgecolor())
        self.bezier_curve = self.axis.add_line(line_bezier)

        self.changeFinished = False
        self._shift_is_held = False
        self._ctrl_is_held = False
        self._alt_is_held = False
        self._index = None  # Active vertex for control point manipulation
        self._index2 = None # Active vertex for Bezier curve dragging
        self._dragAll = False
        self.xclick = None
        self.yclick = None
        self.xp0 = None
        self.yp0 = None

        if len(self.xp)==0:
            self.isFirstDef = True
        else:
            self.isFirstDef = False
            self.control_polygon.set_data(self.xp, self.yp)
            self.canvas.draw()
            self._update_bezier()


    def resetBezier(self, xp, yp):
        """ Set or Reset bezier curve according to input x,y control point coordinates"""
        self.xp = xp
        self.yp = yp
        self.control_polygon.set_data(self.xp, self.yp)
        self.canvas.draw()
        self._update_bezier()


    def on_button_press(self, event):
        """ Control what happens when mouse button is pressed """
        ## Ignore clicks outside axes
        if event.inaxes != self.axis: return
        self.xclick = event.xdata
        self.yclick = event.ydata
        self.xp0 = self.xp
        self.yp0 = self.yp

        res, ind = self.control_polygon.contains(event)

        if res:
            self._index = int(ind['ind'][0])
            if self._index == len(self.xp) - 1:
                self._index = -1
            if not(self._index in self.fixPoints):
                # self._index = int(ind['ind'][0])
                clickOffset = np.sqrt((self.xclick-self.xp[self._index])**2 + (self.yclick-self.yp[self._index])**2)
                if clickOffset<0.02 and self._ctrl_is_held:
                    self._remove_point(event)
                elif clickOffset>0.02  and self._ctrl_is_held:
                    self._add_point(event)
                elif clickOffset<0.02:
                    pass
                else:
                    self._index = None
            else:
                self._index = None
        elif self._ctrl_is_held:
            self._add_point(event)

        res2, ind2 = self.bezier_curve.contains(event)
        if res2 and self.lineDrag:
            self._index2  = int(ind2['ind'][0])


    def on_button_release(self, event):
        """ Control what happens when mouse button is released """
        if event.button != 1: return
        self._index = None
        self._index2 = None
        self._dragAll = False

        if not self._ctrl_is_held:
            self.changeFinished = True
        self._update_bezier()

    def on_key_press(self, event):
        """ Control what happens when keyboard button is pressed """
        if event.key == 'control':
            self._ctrl_is_held = True
            self.changeFinished = False


    def on_key_release(self, event):
        """ Control what happens when keyboard button is released """
        if event.key == 'control':
            self._ctrl_is_held = False
            if len(self.xp)>0:
                self.isFirstDef = False
        self.changeFinished = True
        self._update_bezier()

    def on_motion_notify(self, event):
        """ Control how control points are dragged """
        if self._ctrl_is_held:
            return
        if event.inaxes != self.axis: return
        if self._index is not None:
            x, y = event.xdata, event.ydata
            if self._dragAll:
                self.xp=np.ndarray.tolist(self.xp0+(x-self.xclick))
                self.yp=np.ndarray.tolist(self.yp0+(y-self.yclick))
            else:
                self.xp[self._index] = x
                self.yp[self._index] = y
        elif self._index2 is not None:
            x, y = event.xdata, event.ydata
            if 1:
                self.xp=np.ndarray.tolist(self.xp0+(x-self.xclick))
                self.yp=np.ndarray.tolist(self.yp0+(y-self.yclick))
        else:
            return
        self.control_polygon.set_data(self.xp, self.yp)
        self.changeFinished = False
        self._update_bezier()


    def _add_point(self, event):
        """ Add contrl point """
        if self._index is None and self.isFirstDef:
            self.xp.append(event.xdata)
            self.yp.append(event.ydata)
        elif self._index is None and not self.isFirstDef:
            pass
        else:
            self.xp.insert(self._index+1,event.xdata)
            self.yp.insert(self._index+1,event.ydata)
            self.changeFinished = True
        self.control_polygon.set_data(self.xp, self.yp)

        ## Rebuild Bezier curve and update canvas
        self._update_bezier()


    def _remove_point(self, event):
        """ Remove control point """
        if self._index is not None:
            self.xp.pop(self._index)
            self.yp.pop(self._index)
            self.control_polygon.set_data(self.xp, self.yp)

            ## Rebuild Bezier curve and update canvas
            self.changeFinished = True
            self._update_bezier()


    def _build_bezier(self):
        """ Call self.Bezier, return x,y coordinates of bezier curve """
        x, y = self.Bezier(list(zip(self.xp, self.yp))).T
        return x, y


    def remove(self):
        """ Remove bezier curve (set empty data to hide curve) """
        self.xp = []
        self.yp = []
        self.control_polygon.set_data(self.xp, self.yp)
        self.isRemoved = True
        self.changeFinished = True
        self._update_bezier()



    def _update_bezier(self):
        """ Update bezier curve when control points changed"""
        self.bezier_curve.set_data(self._build_bezier())
        self.canvas.flush_events()
        self.axis.draw_artist(self.bezier_curve)
        self.axis.draw_artist(self.control_polygon)
        self.canvas.update()
        self.canvas.draw()
        self.changed.emit(self.changeFinished)


    def Bernstein(self,n, k):
        """ Calculate Bernstein polynomial """
        coeff = binom(n, k)
        def _bpoly(x):
            return coeff * x ** k * (1 - x) ** (n - k)
        return _bpoly


    def Bezier(self,points):
        """ Build Bezier curve from control points """
        num = self.np
        N = len(points)
        t = np.linspace(0, 1, num=num)
        curve = np.zeros((num, 2))
        for ii in range(N):
            curve += np.outer(self.Bernstein(N - 1, ii)(t), points[ii])
        return curve
