import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np

class Triangle(pg.GraphItem):
    changed = QtCore.pyqtSignal(bool)

    def __init__(self):
        self.dragPoint = None
        self.dragStart = None
        self.dragOffset = None
        self.textItems = []
        pg.GraphItem.__init__(self)


    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.setTexts(self.text)
        self.updateGraph()


    def setTexts(self, text):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        for t in text:
            item = pg.TextItem(t)
            self.textItems.append(item)
            item.setParentItem(self)


    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        for i,item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])


    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            # We are already one step into the drag.
            # Find the point(s) at the mouse cursor when the button was first
            # pressed - correct for :
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragStart = self.data['pos'][ind]
            self.dragOffset = self.data['pos'][ind] - pos*self.data['drag'][ind]

        elif ev.isFinish():
            self.dragPoint = None
            self.changed.emit(True)
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        ind = self.dragPoint.data()[0]
        deltaDrag = ev.pos()*self.data['drag'][ind] -self.dragStart + self.dragOffset
        self.data['pos'][ind] = self.dragStart + deltaDrag

        if ind in self.data['dep']: # Move dependent points accordingly
            for dep in self.data['dep'][ind]:
                self.data['pos'][dep] = self.data['pos'][dep] + deltaDrag

        self.updateGraph()
        ev.accept()