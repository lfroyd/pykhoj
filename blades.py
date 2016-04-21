import numpy as np
from scipy import interpolate, optimize
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from beziercurves import BezierBuilder
from PyQt4 import QtCore


class Blade(QtCore.QObject):

    def __init__(self):

        QtCore.QObject.__init__(self)
        self.avAxis = None
        self.avCanvas = None
        self.accAxis = None
        self.accCanvas = None
        self.rvAxis = None
        self.rvCanvas = None
        self.ghAxis = None
        self.ghCanvas = None
        self.betaAxis = None
        self.betaCanvas = None
        self.energyAxis = None
        self.energyCanvas = None
        self.av = None
        self.avPlot = None
        self.TECurve = None
        self.LECurve = None
        self.streamlines = []
        self.ghPlaneLines = []
        self.radialViewLines = []
        self.np = int(30)



        # self.ns = 10#param('Design Parameters').param('Meridional View').param('Number of streamlines').value()
        # self.D1 = 0.86#params.param('Design Parameters').param('Meridional View').param('Inlet diameter D1').value()
        # self.D2 = 0.5#params.param('Design Parameters').param('Meridional View').param('Outlet diameter D1').value()
        # self.b = 0.17#params.param('Design Parameters').param('Meridional View').param('Shroud height b').value()
        # self.B1 = 0.1#params.param('Design Parameters').param('Meridional View').param('Inlet height B1').value()

    def setPlotAxes(self,axes):
        self.avAxis = axes['Axial View'].axes
        self.avCanvas = self.avAxis.figure.canvas

        self.rvAxis = axes['Radial View'].axes
        self.rvCanvas = self.rvAxis.figure.canvas

        self.ghAxis = axes['GH Plane'].axes
        self.ghCanvas = self.ghAxis.figure.canvas

        self.accAxis = axes['Acceleration'].axes
        self.accCanvas = self.accAxis.figure.canvas

        self.energyAxis = axes['Energy Distribution'].axes
        self.energyCanvas = self.energyAxis.figure.canvas

        self.betaAxis = axes['Beta Distribution'].axes
        self.betaCanvas = self.betaAxis.figure.canvas

        self.blade3DAxis = axes['Blade 3D'].axes
        self.blade3DCanvas = self.blade3DAxis.figure.canvas

        self.initAcceleration()
        self.initBetaDistribution()

    def setDocks(self,docks):
        self.axialViewDock = docks['Axial View']
        self.triangleDock = docks['Triangles']

    def rebuildBlade(self, params):
        self.ns = params.param('Design Parameters').param('Meridional View').param('Number of streamlines').value()
        self.D1 = params.param('Design Parameters').param('Meridional View').param('Inlet diameter D1').value()
        self.D2 = params.param('Design Parameters').param('Meridional View').param('Outlet diameter D2').value()
        self.b = params.param('Design Parameters').param('Meridional View').param('Shroud height b').value()
        self.B1 = params.param('Design Parameters').param('Meridional View').param('Inlet height B1').value()
        self.beta1 = params.param('Design Parameters').param('Reduced Velocities').param('Beta1').value() * np.pi/180
        self.CM1 = params.param('Design Parameters').param('Reduced Velocities').param('CM1_red').value() * params.red
        self.CM2 = params.param('Design Parameters').param('Reduced Velocities').param('CM2_red').value() * params.red
        self.omega = params.param('Design Parameters').param('Machine Data').param('Rotor speed').value() * np.pi/30.


        self.Zmat = np.zeros(shape=(self.np,self.ns))
        self.Rmat = np.zeros(shape=(self.np,self.ns))
        self.Gmat = np.zeros(shape=(self.np, self.ns))
        self.Hmat = np.zeros(shape=(self.np, self.ns))
        self.Bmat = np.zeros(shape=(self.np, self.ns))
        self.CMmat = np.zeros(shape=(self.np, self.ns))
        self.Tmat = np.zeros(shape=(self.np, self.ns))

        self.dbetaRelmat = np.zeros(shape=(self.np, self.ns))

        self.findArea()
        self.calcCM()
        self.setShroud()
        self.setInlet()
        self.calcStreamlines()
        self.truncateBlade()
        self.calcGHplane()
        self.plotBlade3D()
        # print(self.triangleDock.isVisible(),self.axialViewDock.isVisible())


    def calcCM(self):
        for s in range(0,self.ns):
            self.CMmat[:,s] = self.Avect/self.Avect[0]*self.CM1


    def calcStreamlines(self):

        def _secArea(self, s, *data):
            """
            Exact area of streamline cross-section. Used in Scipy.optimize.fsolve
            (zero finding algorithm). Purpose is to determine the streamline
            distance s that yields the correct area A
            """
            R, a, A = data
            return A - np.pi*(2*R-s*np.sin(a))*s

        if 0: # calculates streamlines based on non-linear exact area (slower)
            for s in range(1,self.ns):
                x0 = 0. # initial guess of length of s
                for p in range(1,self.np-1):
                    dr = self.Rmat[p-1,s-1] - self.Rmat[p+1,s-1]
                    dz = self.Zmat[p-1,s-1] - self.Zmat[p+1,s-1]
                    a = np.arctan2(dz,dr)
                    Z = self.Zmat[p,s-1]
                    R = self.Rmat[p,s-1]

                    x = optimize.fsolve(_secArea, x0, args=(R,a,self.Avect[p]))
                    x0 = x # Update guess of length of s as previous correct s

                    self.Rmat[p,s] = R-x*np.sin(a)
                    self.Zmat[p,s] = Z+x*np.cos(a)

                p = self.np-1
                self.Rmat[p,s] = np.sqrt(self.Rmat[p,s-1]**2 - self.Avect[p]/np.pi)
                self.Zmat[p,s] = self.Zmat[p,0]

        else: # calculates streamlines based on approximate area A = 2*pi*r*dr (accurate and fast)
            for s in range(1,self.ns):
                for p in range(1,self.np-1):
                    dr = self.Rmat[p-1,s-1] - self.Rmat[p+1,s-1]
                    dz = self.Zmat[p-1,s-1] - self.Zmat[p+1,s-1]
                    a = np.arctan2(dz,dr)
                    self.Rmat[p,s] = np.sqrt(self.Rmat[p,s-1]**2-self.Avect[p]*np.sin(a)/np.pi)
                    self.Zmat[p,s] = self.Zmat[p,s-1]-(self.Rmat[p,s]-self.Rmat[p,s-1])*np.cos(a)/np.sin(a)
                p = self.np-1
                self.Rmat[p,s] = np.sqrt(self.Rmat[p,s-1]**2 - self.Avect[p]/np.pi)
                self.Zmat[p,s] = self.Zmat[p,0]


    def truncateBlade(self):
        # Insert logic w.r.t blade truncation here
        pass



    def calcGHplane(self):

        self.Bmat[0, :] = self.beta1  # fill leading edge beta angle in beta array
        self.Bmat[-1, :] = np.arctan2(self.CMmat[-1, :], self.omega * self.Rmat[-1, :])

        for s in range(0,self.ns):
            self.Bmat[:, s] = self.Bmat[0, s] - (self.Bmat[0, s] - self.Bmat[-1, s]) * np.asarray(self.betaCurve.bezier_curve.get_ydata())
            for p in range(1,self.np):
                self.Gmat[p,s] = self.Gmat[p-1,s] + np.sqrt((self.Rmat[p-1,s] - self.Rmat[p,s])**2 + (self.Zmat[p-1,s] - self.Zmat[p,s])**2)
                self.Hmat[p,s] = self.Hmat[p-1,s] + (self.Gmat[p,s]-self.Gmat[p-1,s])/np.tan(self.Bmat[p,s])
                self.Tmat[p,s] = self.Tmat[p-1,s] + (self.Hmat[p,s] - self.Hmat[p-1,s])/self.Rmat[p,s]

    def initAcceleration(self):
        xvals, yvals = self.getAccCurve()
        # Initialize empty Line2D object, set initial data and add to figure axis
        line = Line2D([], [], ls='--', c='black', marker='o', mew=2, mec='black')
        line.set_data(xvals, yvals)
        self.accAxis.add_line(line)
        # Create Bezier curve
        self.accCurve = BezierBuilder(line, self.accAxis, self.np, fixpoints=[0,-1])
        self.accCurve.changed.connect(self.accelerationChanged)


    def initBetaDistribution(self):
        xvals, yvals = self.getBetaCurve()
        # Initialize empty Line2D object, set initial data and add to figure axis
        line = Line2D([], [], ls='--', c='black', marker='o', mew=2, mec='black')
        line.set_data(xvals, yvals)
        self.betaAxis.add_line(line)
        # Create Bezier curve
        self.betaCurve = BezierBuilder(line, self.betaAxis, self.np, fixpoints=[0,-1])
        self.betaCurve.changed.connect(self.betaChanged)


    def resetBetaDistribution(self):
        xvals, yvals = self.getBetaCurve()
        # Update Bezier curve
        self.betaCurve.resetBezier(list(xvals),list(yvals))
        self.betaCurve.changed.connect(self.betaChanged)


    def resetAcceleration(self):
        xvals, yvals = self.getAccCurve()
        # Update Bezier curve
        self.accCurve.resetBezier(list(xvals), list(yvals))
        self.accCurve.changed.connect(self.betaChanged)


    def accelerationChanged(self,isChanged):
        if isChanged:
            self.findArea()
            self.calcStreamlines()
            self.updateAxialView()
            self.calcGHplane()
            self.updateGHplane()
            self.updateRadialView()

    def betaChanged(self, isChanged):
        if isChanged:
            self.calcGHplane()
            self.updateGHplane()
            self.updateRadialView()

    def findArea(self):
        A1 = np.pi*self.D1*self.B1
        A2 = np.pi*((0.5*self.D2)**2-0.01*(0.5*self.D2)**2)
        self.dAx = np.asarray(self.accCurve.bezier_curve.get_xdata())
        self.dAy = np.asarray(self.accCurve.bezier_curve.get_ydata())
        self.Avect = (A1*(1-self.dAy)+A2*self.dAy)/float(self.ns-1)


    def setShroud(self):
        if 1: # Ellipse
            theta = np.linspace(np.pi/2., np.pi,self.np)
            R_shroud = 0.5*self.D1 + 0.5*(self.D1-self.D2)*np.cos(theta)
            Z_shroud = self.b*np.sin(theta)

            self.Rmat[:,0] = R_shroud
            self.Zmat[:,0] = Z_shroud

        else: # Bezier curve
            pass


    def setInlet(self):
        if 1:
            z = np.linspace(self.b, self.b+self.B1, self.ns)
            self.Rmat[0,:] = 0.5*self.D1
            self.Zmat[0,:] = z


    def distributeEvenly(self):
        # dR = R_shroud[1:]-R_shroud[:-1]
        # dZ = Z_shroud[1:]-Z_shroud[:-1]
        # ds = (np.sqrt(dR**2 + dZ**2))
        # s =np.zeros(self.np)
        # s[1::] = np.cumsum(ds)
        # r = np.sqrt(R_shroud**2 + Z_shroud**2)
        # Theta=np.arctan(Z_shroud/-R_shroud)
        # S=np.linspace(0,max(s),self.np)
        # # print Theta.shape, S.shape, s.shape, ds.shape,
        # pchip1 = interpolate.pchip(s,Theta)
        # THeta = pchip1(S)
        # pchip2 = interpolate.pchip(s,Theta)
        # R = pchip2(S)
        #
        # self.Rmat[:,0] = -R*np.cos(THeta)
        # self.Zmat[:,0] = R*np.sin(THeta)
        pass


    def updateAxialView(self):
        for s in range(0,min(self.ns,len(self.streamlines))):
            self.streamlines[s].set_data(self.Rmat[:,s], self.Zmat[:,s])
        for s in range(min(self.ns,len(self.streamlines)),self.ns):
            self.streamlines.append(Line2D([self.Rmat[:,s]], [self.Zmat[:,s]]))
            self.avAxis.add_line(self.streamlines[s])
        for s in range(self.ns,len(self.streamlines)):
            self.streamlines[s].set_data([], [])
        self.avAxis.set_ylim([0,(self.B1+self.b)*1.2])
        self.avAxis.set_xlim([0,self.D1*0.5*1.2])
        self.avCanvas.flush_events()
        self.avCanvas.update()
        self.avCanvas.draw()


    def updateGHplane(self):
        for s in range(0, min(self.ns, len(self.ghPlaneLines))):
            self.ghPlaneLines[s].set_data(self.Gmat[:, s], self.Hmat[:, s])
        for s in range(min(self.ns, len(self.ghPlaneLines)), self.ns):
            self.ghPlaneLines.append(Line2D([self.Gmat[:, s]], [self.Hmat[:, s]]))
            self.ghAxis.add_line(self.ghPlaneLines[s])
        for s in range(self.ns, len(self.ghPlaneLines)):
            self.ghPlaneLines[s].set_data([], [])
        self.ghCanvas.flush_events()
        self.ghCanvas.update()
        self.ghCanvas.draw()


    def updateRadialView(self):
        for s in range(0, min(self.ns, len(self.radialViewLines))):
            self.radialViewLines[s].set_data(self.Tmat[:, s], self.Rmat[:, s])
        for s in range(min(self.ns, len(self.radialViewLines)), self.ns):
            self.radialViewLines.append(Line2D([self.Tmat[:, s]], [self.Rmat[:, s]]))
            self.rvAxis.add_line(self.radialViewLines[s])
        for s in range(self.ns, len(self.radialViewLines)):
            self.radialViewLines[s].set_data([], [])
        self.rvAxis.set_rmax(self.D1 * 0.5)
        self.rvCanvas.flush_events()
        self.rvCanvas.update()
        self.rvCanvas.draw()


    def plotBlade3D(self):
        x = self.Rmat * np.cos(self.Tmat)
        y = self.Rmat * np.sin(self.Tmat)
        self.blade3DAxis.clear() # there's no set_data() method for 3D collections so we have to clear axis instead
        self.blade3DAxis.plot_surface(x, y, self.Zmat, rstride=1, cstride=1)
        self.blade3DAxis.auto_scale_xyz
        self.blade3DCanvas.flush_events()
        self.blade3DCanvas.update()
        self.blade3DCanvas.draw()


    def setBlade3DProjection(self,projection):
        print(projection)


    def setCascade3DProjection(self, projection):
        print(projection)


    def addTECurve(self, isChecked):
        if self.LECurve is not None:
            self.LECurve.isFirstDef = False

        if isChecked: # add new curve when checked
            xvals, yvals = self.getTECurve()
            # Initialize empty Line2D object, set initial data and add to figure axis
            line = Line2D([], [], ls='--', c='#666666', marker='o', mew=2, mec='#204a87')
            line.set_data(xvals,yvals)
            self.avAxis.add_line(line)

            # Create Bezier curve
            self.TECurve = BezierBuilder(line,self.avAxis,200)

        else: # remove curve when unchecked
            self.TECurve.remove()


    def addLECurve(self, isChecked):
        if self.TECurve is not None:
            self.TECurve.isFirstDef = False

        if isChecked: # add new curve when checked
            xvals, yvals = self.getLECurve()
            # Initialize empty Line2D object, set initial data and add to figure axis
            line = Line2D([], [], ls='--', c='red', marker='o', mew=2, mec='red')
            line.set_data(xvals,yvals)
            self.avAxis.add_line(line)

            # Create Bezier curve
            self.LECurve = BezierBuilder(line,self.avAxis, 200)

        else:# remove curve when unchecked
            self.LECurve.remove()


    def getLECurve(self):
        return [],[]


    def getTECurve(self):
        return [],[]


    def getAccCurve(self):
        xvals = np.array([0., 0.25, 0.50, 0.75, 1.])
        yvals = np.array([0., 0.60, 0.85, 0.98, 1.])
        return xvals, yvals


    def getBetaCurve(self):
        xvals = np.array([0., 0.25, 0.50, 0.75, 1.])
        yvals = np.array([0., 0.25, 0.50, 0.75, 1.])
        return xvals, yvals

    # def spacedMarks(x, y, Nmarks):
    #     import scipy.integrate
    #     #
    #     # if data_ratio is None:
    #     #     data_ratio = plt.gca().get_data_ratio()
    #
    #     dydx = np.gradient(y, x[1])
    #     dxdx = np.gradient(x, x[1])*0.01
    #     arclength = scipy.integrate.cumtrapz(np.sqrt(dydx**2 + dxdx**2), x, initial=0)
    #     marks = np.linspace(0, max(arclength), Nmarks)
    #     markx = np.interp(marks, arclength, x)
    #     marky = np.interp(markx, x, y)
    #     return markx, marky



if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    b = Blade(ax)

    p1 = 0
    if p1 == 0:
        for i in range(0,b.ns):
            ax.plot(b.Rmat[:,i],b.Zmat[:,i],color='blue')
            # inlet = ax.plot(b.Rmat[:,1],b.Zmat[:,1],'o',color='red')
            # inlet = ax.plot(b.Rmat[:,2],b.Zmat[:,2],'o',color='green')
    elif p1 == 1:
        ax = fig.add_subplot(111)
        ax.plot(b.dAx,b.Avect)



    plt.show()

