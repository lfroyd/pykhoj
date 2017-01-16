import numpy as np
from scipy import interpolate, optimize, integrate
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from beziercurves import BezierBuilder
from PyQt4 import QtCore


class Blade(QtCore.QObject):

    def __init__(self):

        QtCore.QObject.__init__(self)
        self.avPlot = None
        self.TECurve = None
        self.LECurve = None
        self.accCurve = None
        self.betaCurve = None
        self.streamlines = []
        self.streamlinesCut =[]
        self.streamlinesCut2 = []
        self.ghPlaneLines = []
        self.radialViewLines = []
        self.TEPoints = []
        self.LEPoints = []
        self.npa = int(30)
        self.np = int(30)
        self.blade3D = {}
        self.av = {}
        self.rv ={}
        self.gh = {}
        self.beta = {}
        self.energy = {}
        self.acc = {}

        self.Zmat = None
        self.Rmat = None
        self.Gmat = None
        self.Hmat = None
        self.Bmat = None
        self.CMmat = None
        self.Tmat = None
        self.Amat = None

        self.colorDict = {'Default': None,
                     'Z-value': self.Zmat,
                     'Radius': self.Rmat,
                     'CM velocity': self.CMmat,
                     'Theta angle': self.Tmat,
                     'Beta angle': self.Bmat}
        self.Smat = None
        self.blade3D['Colour'] = None


    def setPlotAxes(self,dockWidgetDict):
        """ Define instance variables holding the reference to each graphic plotting area """
        self.av['Axis'] = dockWidgetDict['Axial View'][0].axes
        self.av['Canvas'] = self.av['Axis'].figure.canvas

        self.rv['Axis'] = dockWidgetDict['Radial View'][0].axes
        self.rv['Canvas'] = self.rv['Axis'].figure.canvas

        self.gh['Axis'] = dockWidgetDict['GH Plane'][0].axes
        self.gh['Canvas'] = self.gh['Axis'].figure.canvas

        self.acc['Axis'] = dockWidgetDict['Acceleration'][0].axes
        self.acc['Canvas'] = self.acc['Axis'].figure.canvas

        self.energy['Axis'] = dockWidgetDict['Energy Distribution'][0].axes
        self.energy['Canvas'] = self.energy['Axis'].figure.canvas

        self.beta['Axis'] = dockWidgetDict['Beta Distribution'][0].axes
        self.beta['Canvas'] = self.beta['Axis'].figure.canvas

        self.blade3D['Axis'] = dockWidgetDict['Blade 3D'][0].axes
        self.blade3D['Canvas'] = self.blade3D['Axis'].figure.canvas


    def readParams(self, params):
        """ Update local instance variables holding designparameters """
        ## update values from parameter tree
        self.ns = params.param('Design Parameters').param('Meridional View').param('Number of streamlines').value()
        self.D1 = params.param('Design Parameters').param('Meridional View').param('Inlet diameter D1').value()
        self.D2 = params.param('Design Parameters').param('Meridional View').param('Outlet diameter D2').value()
        self.b = params.param('Design Parameters').param('Meridional View').param('Shroud height b').value()
        self.B1 = params.param('Design Parameters').param('Meridional View').param('Inlet height B1').value()
        self.beta1 = params.param('Design Parameters').param('Reduced Velocities').param('Beta1').value() * np.pi/180
        self.CM1 = params.param('Design Parameters').param('Reduced Velocities').param('CM1_red').value() * params.red
        self.CM2 = params.param('Design Parameters').param('Reduced Velocities').param('CM2_red').value() * params.red
        self.omega = params.param('Design Parameters').param('Machine Data').param('Rotor speed').value() * np.pi/30.

    def rebuildBlade(self):
        """ Update the blade and various views when changes are made in design parameters """

        ## reset meridional view arrays (mat's)
        self.ZAmat = np.zeros(shape=(self.npa,self.ns))
        self.RAmat = np.zeros(shape=(self.npa,self.ns))
        self.CMAmat = np.zeros(shape=(self.npa, self.ns))
        self.SAmat = np.zeros(shape=(self.npa, self.ns))
        self.AAmat = np.zeros(shape=(self.npa, self.ns))

        ## reset blade definition arrays (mat's)
        self.Zmat = np.zeros(shape=(self.np, self.ns))
        self.Rmat = np.zeros(shape=(self.np, self.ns))
        self.Gmat = np.zeros(shape=(self.np, self.ns))
        self.Hmat = np.zeros(shape=(self.np, self.ns))
        self.Bmat = np.zeros(shape=(self.np, self.ns))
        self.Tmat = np.zeros(shape=(self.np, self.ns))
        self.Amat = np.zeros(shape=(self.np, self.ns))
        self.CMmat = np.zeros(shape=(self.np, self.ns))
        self.Smat = np.zeros(shape=(self.np, self.ns))

        self.findArea()
        # self.calcCM()
        self.setShroud()
        self.setInlet()
        self.calcStreamlines()
        self.truncateBlade()
        self.updateAxialView()
        self.updateAxialViewCut()
        self.calcGHplane()
        self.plotBlade3D()
        # print(self.triangleDock.isVisible(),self.axialViewDock.isVisible())


    def _secArea(self, s, *data):
        """ Exact area of streamline cross-section. Used in Scipy.optimize.fsolve (zero finding algorithm).
        Purpose is to determine the streamline distance s that yields the exact area A """
        R, a, A = data
        return A - np.pi * (2 * R - s * np.sin(a)) * s


    def calcStreamlines(self):
        """ Calculate streamlines from shroud towards hub based on:
            spanwise (shroud to hub): equal distribution of flow in each streamline
            chordwise (LE to TE): acceleration distribution defined by acceleration control bezier curve
        """

        dr = np.gradient(self.RAmat[:,0])
        dzdr = np.gradient(self.ZAmat[:,0],dr)
        drdr = np.gradient(self.RAmat[:,0],dr)
        self.SAmat[:,0] = abs(integrate.cumtrapz(np.sqrt(dzdr ** 2 + drdr ** 2), self.RAmat[:,0], initial=0))
        self.AAmat[:,0] = np.interp(self.SAmat[:,0]/self.SAmat[-1,0],self.dAx,self.Avect)
        self.CMAmat[:,0] = self.AAmat[:,0] / self.AAmat[0,0] * self.CM1

        if False: # calculates streamlines based on non-linear exact area (slower)
            ## TODO: Method is not exposed to user selection, consider to remove or expose
            for s in range(1,self.ns):
                x0 = 0. # initial guess of length of s
                for p in range(1,self.npa-1):
                    dr = self.RAmat[p-1,s-1] - self.RAmat[p+1,s-1]
                    dz = self.ZAmat[p-1,s-1] - self.ZAmat[p+1,s-1]
                    a = np.arctan2(dz,dr)
                    Z = self.ZAmat[p,s-1]
                    R = self.RAmat[p,s-1]
                    # print R,a,self.AAmat[p,0]
                    x = optimize.fsolve(self._secArea, x0, args=(R,a,self.AAmat[p,0])) # self.AAmat[p,s-1]
                    x0 = x # Update guess of length of s as previous correct s

                    self.RAmat[p,s] = R-x*np.sin(a)
                    self.ZAmat[p,s] = Z+x*np.cos(a)

                ## Add last point on streamline s
                p = self.npa-1
                self.RAmat[p,s] = np.sqrt(self.RAmat[p,s-1]**2 - self.AAmat[p,0]/np.pi)
                self.ZAmat[p,s] = self.ZAmat[p,0]

                ## Calculate length of streamline s
                dr = np.gradient(self.RAmat[:,s])
                dzdr = np.gradient(self.ZAmat[:,s], dr)
                drdr = np.gradient(self.RAmat[:,s], dr)
                self.SAmat[:, s] = abs(integrate.cumtrapz(np.sqrt(dzdr ** 2 + drdr ** 2), self.RAmat[:,s], initial=0))
                self.AAmat[:, s] = np.interp(self.SAmat[:,s] / self.SAmat[-1,s], self.dAx, self.Avect)
                self.CMAmat[:, s] = self.AAmat[:,s] / self.AAmat[0,s] * self.CM1

        else: # calculates streamlines based on approximate area A = 2*pi*r*dr (quite accurate and very fast)
            for s in range(1,self.ns):
                for p in range(1,self.npa-1):
                    dr = self.RAmat[p-1,s-1] - self.RAmat[p+1,s-1]
                    dz = self.ZAmat[p-1,s-1] - self.ZAmat[p+1,s-1]
                    a = np.arctan2(dz,dr)
                    self.RAmat[p,s] = np.sqrt(self.RAmat[p,s-1]**2-self.AAmat[p,0]*np.sin(a)/np.pi)
                    self.ZAmat[p,s] = self.ZAmat[p,s-1]-(self.RAmat[p,s]-self.RAmat[p,s-1])*np.cos(a)/np.sin(a)
                    # self.SAmat[p,s] = self.SAmat[p-1,s] + np.sqrt(self.S)

                ## Add last point on streamline s
                p = self.npa - 1
                self.RAmat[p, s] = np.sqrt(self.RAmat[p, s - 1] ** 2 - self.AAmat[p, 0] / np.pi)
                self.ZAmat[p, s] = self.ZAmat[p, 0]

                ## Calculate length of streamline s
                dr = np.gradient(self.RAmat[:, s])
                dzdr = np.gradient(self.ZAmat[:, s], dr)
                drdr = np.gradient(self.RAmat[:, s], dr)
                self.SAmat[:, s] = abs(integrate.cumtrapz(np.sqrt(dzdr ** 2 + drdr ** 2), self.RAmat[:, s], initial=0))
                self.AAmat[:, s] = np.interp(self.SAmat[:, s] / self.SAmat[-1, s], self.dAx, self.Avect)
                self.CMAmat[:, s] = self.AAmat[:, s] / self.AAmat[0, s] * self.CM1


    def truncateBlade(self):
        """ Define logic w.r.t. truncation of blade defined by LE and TE bezier control curves """

        ## Find intersection between Trailing Edge (TE) and streamlines
        end_index = np.zeros(self.ns, dtype=int)+ int(self.npa)
        if self.TECurve is not None:
            if not self.TECurve.isRemoved:
                r2 = np.empty(self.ns)
                z2 = np.empty(self.ns)
                TEr = np.asarray(self.TECurve.bezier_curve.get_xdata())
                TEz = np.asarray(self.TECurve.bezier_curve.get_ydata())
                for s in range(0, int(self.ns)):
                    tmp = self.find_intersect_vec(self.RAmat[:,s], self.ZAmat[:,s], TEr, TEz)
                    r2[s] = tmp[0,0]
                    z2[s] = tmp[0,1]
                    end_index[s] = self.npa - np.searchsorted(self.ZAmat[:,s][::-1], z2[s], side='left')
                    self.ZAmat[end_index[s],s] = z2[s]
                    self.RAmat[end_index[s],s] = r2[s]

        ## Find intersection between Leading Edge (LE) and streamlines
        start_index = np.zeros(self.ns, dtype=int)
        if self.LECurve is not None:
            if not self.LECurve.isRemoved:
                r1 = np.empty(self.ns)
                z1 = np.empty(self.ns)
                LEr = np.asarray(self.LECurve.bezier_curve.get_xdata())
                LEz = np.asarray(self.LECurve.bezier_curve.get_ydata())
                for s in range(0, int(self.ns)):
                    tmp = self.find_intersect_vec(self.RAmat[:, s], self.ZAmat[:, s], LEr, LEz)
                    r1[s] = tmp[0,0]
                    z1[s] = tmp[0,1]
                    start_index[s] = int(np.searchsorted(self.D1*0.5-self.RAmat[:,s], self.D1*0.5-r1[s], side='right'))
                    self.ZAmat[start_index[s],s] = z1[s]
                    self.RAmat[start_index[s],s] = r1[s]

        ## Truncate blade between LE and TE and redistribute points on streamlines in between
        ## Uses SciPy Interpolate to curve fit a 2D B-spline and distribute points on the spline curve
        ## The B-spline representation is fast, but does not always perfectly follow the original streamlines
        ## TODO: Maybe better fit could be achieved by tweaking the options, e.g. spline order?
        ## TODO: It would be good to add alternative approaches to this later, for the user to select
        for s in range(0, int(self.ns)):

            data = np.vstack((self.RAmat[start_index[s]:end_index[s]+1,s], self.ZAmat[start_index[s]:end_index[s]+1,s]))
            tck, u = interpolate.splprep(data)
            new = interpolate.splev(np.linspace(0, 1, self.np), tck)
            self.Rmat[:,s] = new[0]
            self.Zmat[:,s] = new[1]

            ## Calculate length of streamline s
            dr = np.gradient(self.Rmat[:, s])
            dz = np.gradient(self.Zmat[:, s])
            dzdr = np.gradient(self.Zmat[:, s], dr)
            drdr = np.gradient(self.Rmat[:, s], dr)
            self.Smat[:, s] = abs(integrate.cumtrapz(np.sqrt(dzdr ** 2 + drdr ** 2), self.Rmat[:, s], initial=0))

            self.Amat[:,s] = np.interp(self.Smat[:, s], self.SAmat[:,s], self.AAmat[:,s])
            self.CMmat[:, s] = np.interp(self.Smat[:, s], self.SAmat[:, s], self.CMAmat[:, s])


    def find_intersect_vec(self, x1, y1, x2, y2):
        """
        Find intersection points between curves defined by [x1,y1] and [x2,y2]
        Source: Answer by Jaime (answered Jul 29 '13 at 18:46) from
        https://stackoverflow.com/questions/17928452/find-all-intersections-of-xy-data-point-graph-with-numpy
        """
        p = np.column_stack((x1, y1))
        q = np.column_stack((x2, y2))
        p0, p1, q0, q1 = p[:-1], p[1:], q[:-1], q[1:]
        rhs = q0 - p0[:, np.newaxis, :]
        mat = np.empty((len(p0), len(q0), 2, 2))
        mat[..., 0] = (p1 - p0)[:, np.newaxis]
        mat[..., 1] = q0 - q1
        mat_inv = -mat.copy()
        mat_inv[..., 0, 0] = mat[..., 1, 1]
        mat_inv[..., 1, 1] = mat[..., 0, 0]
        det = mat[..., 0, 0] * mat[..., 1, 1] - mat[..., 0, 1] * mat[..., 1, 0]
        mat_inv /= det[..., np.newaxis, np.newaxis]
        import numpy.core.umath_tests as ut
        params = ut.matrix_multiply(mat_inv, rhs[..., np.newaxis])
        intersection = np.all((params >= 0) & (params <= 1), axis=(-1, -2))
        p0_s = params[intersection, 0, :] * mat[intersection, :, 0]
        return p0_s + p0[np.where(intersection)[0]]


    def calcGHplane(self):
        """ Calculate the blade 3D shape using the G-H plane notation using either:
            Beta angle distribution: according to beta control bezier curve
            Energy distribution: according to energy (UCu) control bezier curve
        """
        ## TODO: add energy distribution logic

        ## Fill leading edge beta angle in beta array
        ## TODO: inlet cutting should probably change the inlet beta
        self.Bmat[0, :] = self.beta1

        ## Fill trailig edge beta angle in beta array.
        self.Bmat[-1, :] = np.arctan2(self.CMmat[-1, :], self.omega * self.Rmat[-1, :])

        ## For each streamline distribute the change of beta from LE to TE according to beta bezier control curve
        for s in range(0,self.ns):
            self.Bmat[:, s] = self.Bmat[0, s] - (self.Bmat[0, s] - self.Bmat[-1, s]) * np.asarray(self.betaCurve.bezier_curve.get_ydata())
            ## For each point on streamline calculate G-H coordinates and radial angle Theta accordingly
            ## TODO: inlet cutting should imply that inlet Theta angle is no longer zero. Logic to be developed
            for p in range(1,self.np):
                self.Gmat[p,s] = self.Gmat[p-1,s] + np.sqrt((self.Rmat[p-1,s] - self.Rmat[p,s])**2 + (self.Zmat[p-1,s] - self.Zmat[p,s])**2)
                self.Hmat[p,s] = self.Hmat[p-1,s] + (self.Gmat[p,s]-self.Gmat[p-1,s])/np.tan(self.Bmat[p,s])
                self.Tmat[p,s] = self.Tmat[p-1,s] + (self.Hmat[p,s] - self.Hmat[p-1,s])/self.Rmat[p,s]


    def initAcceleration(self):
        """ Initialize acceleration bezier control curve """
        xvals, yvals = self.getAccCurve()
        # Initialize empty Line2D object, set initial data and add to figure axis
        line = Line2D([], [], ls='--', c='black', marker='o', mew=2, mec='black')
        line.set_data(xvals, yvals)
        self.acc['Axis'].add_line(line)
        # Create Bezier curve
        self.accCurve = BezierBuilder(line, self.acc['Axis'], 50, fixpoints=[0,-1])
        self.accCurve.changed.connect(self.accelerationChanged)


    def initBetaDistribution(self):
        """ Initialize beta angle bezier control curve """
        xvals, yvals = self.getBetaCurve()
        # Initialize empty Line2D object, set initial data and add to figure axis
        line = Line2D([], [], ls='--', c='black', marker='o', mew=2, mec='black')
        line.set_data(xvals, yvals)
        self.beta['Axis'].add_line(line)
        # Create Bezier curve
        self.betaCurve = BezierBuilder(line, self.beta['Axis'], self.np, fixpoints=[0,-1])
        self.betaCurve.changed.connect(self.betaChanged)


    def resetBetaDistribution(self):
        """ Reset beta angle bezier control curve """
        xvals, yvals = self.getBetaCurve()
        # Update Bezier curve
        self.betaCurve.resetBezier(list(xvals),list(yvals))


    def resetAcceleration(self):
        """ Reset acceleration bezier control curve """
        xvals, yvals = self.getAccCurve()
        # Update Bezier curve
        self.accCurve.resetBezier(list(xvals), list(yvals))


    def accelerationChanged(self,isChanged):
        """ Update blade and various views when acceleration control curve changed """
        if isChanged:
            self.rebuildBlade()


    def bladeCutChanged(self, isChanged):
        """ Update blade and various views when LE/TE control curve changed """
        if isChanged:
            self.rebuildBlade()


    def betaChanged(self, isChanged):
        """ Update blade and various views when beta angle control curve changed """
        if isChanged:
            self.calcGHplane()
            self.updateGHplane()
            self.updateRadialView()
            self.plotBlade3D()


    def findArea(self):
        """ Find chordwise (LE to TE) distribution of meridional cross-section area according to
            acceleration control curve
        """
        A1 = np.pi*self.D1*self.B1
        A2 = np.pi*((0.5*self.D2)**2-0.01*(0.5*self.D2)**2)
        self.dAx = np.asarray(self.accCurve.bezier_curve.get_xdata())
        self.dAy = np.asarray(self.accCurve.bezier_curve.get_ydata())
        self.Avect = (A1*(1-self.dAy)+A2*self.dAy)/float(self.ns-1)


    def setShroud(self):
        """ Set radial (R) and axial (Z) coordinates of shroud (defined in meridional projection) """
        if 1: # Ellipse
            theta = np.linspace(np.pi/2., np.pi,self.npa)
            R_shroud = 0.5*self.D1 + 0.5*(self.D1-self.D2)*np.cos(theta)
            Z_shroud = self.b*np.sin(theta)

            self.RAmat[:,0] = R_shroud
            self.ZAmat[:,0] = Z_shroud

        else: # Bezier curve
            pass


    def setInlet(self):
        """ Set radial (R) and axial (Z) coordinates of inlet (defined in meridional projection) """
        if 1:
            z = np.linspace(self.b, self.b+self.B1, self.ns)
            self.RAmat[0,:] = 0.5*self.D1
            self.ZAmat[0,:] = z


    def updateAxialView(self):
        """ Update axial view (meridional projection view) when design params changed """
        ## For streamlines already defined use set_data to update line objects:
        for s in range(0,min(self.ns,len(self.streamlines))):
            self.streamlines[s].set_data(self.RAmat[:,s], self.ZAmat[:,s])
        ## For new streamlines append new line objects:
        for s in range(min(self.ns,len(self.streamlines)),self.ns):
            self.streamlines.append(Line2D([self.RAmat[:,s]], [self.ZAmat[:,s]],color='0.80'))
            self.av['Axis'].add_line(self.streamlines[s])
        ## For superfluous streamlines already defined set empty data using set_data to hide line objects:
        for s in range(self.ns,len(self.streamlines)):
            self.streamlines[s].set_data([], [])

        # self.av['Axis'].set_ylim([0,(self.B1+self.b)*1.2])
        # self.av['Axis'].set_xlim([0,self.D1*0.5*1.2])
        # self.av['Canvas'].flush_events()
        # self.av['Canvas'].update()
        # self.av['Canvas'].draw()


    def updateAxialViewCut(self):
        """ Update blade section in axial view (meridional projection view) when LE/TE control curves changed """
        ## TODO: When axial view changes LE/TE control curves should be moved accordingly  - logic to be developed
        for s in range(0, min(self.ns, len(self.streamlinesCut))):
            self.streamlinesCut[s].set_data(self.Rmat[:, s], self.Zmat[:, s])
        for s in range(min(self.ns, len(self.streamlinesCut)), self.ns):
            self.streamlinesCut.append(Line2D([self.Rmat[:, s]], [self.Zmat[:, s]],linestyle='-', color='k',marker='.'))
            self.av['Axis'].add_line(self.streamlinesCut[s])
        for s in range(self.ns, len(self.streamlinesCut)):
            self.streamlinesCut[s].set_data([], [])

        self.av['Axis'].set_ylim([0, (self.B1 + self.b) * 1.2])
        self.av['Axis'].set_xlim([0, self.D1 * 0.5 * 1.2])
        self.av['Canvas'].flush_events()
        self.av['Canvas'].update()
        self.av['Canvas'].draw()


    def updateGHplane(self):
        """ Update GH-plane view when something changed """
        for s in range(0, min(self.ns, len(self.ghPlaneLines))):
            self.ghPlaneLines[s].set_data(self.Gmat[:, s], self.Hmat[:, s])
        for s in range(min(self.ns, len(self.ghPlaneLines)), self.ns):
            self.ghPlaneLines.append(Line2D([self.Gmat[:, s]], [self.Hmat[:, s]]))
            self.gh['Axis'].add_line(self.ghPlaneLines[s])
        for s in range(self.ns, len(self.ghPlaneLines)):
            self.ghPlaneLines[s].set_data([], [])
        self.gh['Canvas'].flush_events()
        self.gh['Canvas'].update()
        self.gh['Canvas'].draw()


    def updateRadialView(self):
        """ Update Radial view when something changed """
        for s in range(0, min(self.ns, len(self.radialViewLines))):
            self.radialViewLines[s].set_data(self.Tmat[:, s], self.Rmat[:, s])
        for s in range(min(self.ns, len(self.radialViewLines)), self.ns):
            self.radialViewLines.append(Line2D([self.Tmat[:, s]], [self.Rmat[:, s]]))
            self.rv['Axis'].add_line(self.radialViewLines[s])
        for s in range(self.ns, len(self.radialViewLines)):
            self.radialViewLines[s].set_data([], [])
        self.rv['Axis'].set_rmax(self.D1 * 0.5)
        self.rv['Canvas'].flush_events()
        self.rv['Canvas'].update()
        self.rv['Canvas'].draw()

    def getCurvature(self):
        pass
        # H1,K1 = self.surfature(x,y,self.Zmat)
        # K2 = self.gaussian_curvature(self.Zmat)
        # N = K/K.max()
        # Gx, Gy = np.gradient(self.Zmat)  # gradients with respect to column and row
        # G = (Gx ** 2 + Gy ** 2) ** .5  # gradient magnitude
        # N = G #/ G.max()  # normalize 0..1
        # print K1.shape, K2.shape, x.shape, N.shape


    def plotBlade3D(self):
        """ Plot 3D projection in Blade 3D graphic view """
        x = self.Rmat * np.cos(self.Tmat)
        y = self.Rmat * np.sin(self.Tmat)

        self.blade3D['Axis'].clear()  # there's no set_data() method for 3D collections so we have to clear axis instead
        if self.blade3D['Colour'] is None:
            self.blade3D['Axis'].plot_surface(x, y, self.Zmat, rstride=1, cstride=1, alpha=0.5, linewidth=0)
        else:
            N = eval('self.' + self.blade3D['Colour'])
            N = N/N.max()
            self.blade3D['Axis'].plot_surface(x, y, self.Zmat, rstride=1, cstride=1, alpha=0.5, linewidth = 0, facecolors=cm.jet(N))

        if not self.blade3D['showAxis']:
            self.blade3D['Axis'].set_axis_off()

        self.blade3D['Canvas'].flush_events()
        self.blade3D['Canvas'].update()
        self.blade3D['Canvas'].draw()


    def plotCascade3D(self):
        """ Plot 3D projection in Blade Cascade (Turbine) 3D graphic view """
        x = self.Rmat * np.cos(self.Tmat)
        y = self.Rmat * np.sin(self.Tmat)

        self.blade3D['Axis'].clear()  # there's no set_data() method for 3D collections so we have to clear axis instead
        self.blade3D['Axis'].plot_surface(x, y, self.Zmat, rstride=1, cstride=1, alpha=0.7)
        # self.blade3D['Axis'].auto_scale_xyz
        self.blade3D['Canvas'].flush_events()
        self.blade3D['Canvas'].update()
        self.blade3D['Canvas'].draw()


    def addTECurve(self, isChecked):
        """ Add or remove Trailing Edge bezier control curve"""
        if self.LECurve is not None:
            self.LECurve.isFirstDef = False

        if isChecked: # add new curve when checked
            xvals, yvals = self.getTECurve()
            # Initialize empty Line2D object, set initial data and add to figure axis
            line = Line2D([], [], ls='--', c='#666666', marker='o', mew=2, mec='#204a87')
            line.set_data(xvals,yvals)
            self.av['Axis'].add_line(line)

            # Create Bezier curve
            self.TECurve = BezierBuilder(line,self.av['Axis'],200,linedrag=True,fixpoints=[])
            self.TECurve.changed.connect(self.bladeCutChanged)

        else: # remove curve when unchecked
            self.TECurve.remove()
            self.TECurve = None


    def addLECurve(self, isChecked):
        """ Add or remove Leading Edge bezier control curve"""
        if self.TECurve is not None:
            self.TECurve.isFirstDef = False

        if isChecked: # add new curve when checked
            xvals, yvals = self.getLECurve()
            # Initialize empty Line2D object, set initial data and add to figure axis
            line = Line2D([], [], ls='--', c='red', marker='o', mew=2, mec='red')
            line.set_data(xvals,yvals)
            self.av['Axis'].add_line(line)

            # Create Bezier curve
            self.LECurve = BezierBuilder(line,self.av['Axis'],200,linedrag=True,fixpoints=[])
            self.LECurve.changed.connect(self.bladeCutChanged)

        else:# remove curve when unchecked
            self.LECurve.remove()
            self.LECurve = None


    def getLECurve(self):
        """ Define default control points for Leading Edge bezier control curve"""
        return [],[]


    def getTECurve(self):
        """ Define default control points for Trailing Edge bezier control curve"""
        return [],[]


    def getAccCurve(self):
        """ Define default control points for Acceleration bezier control curve"""
        xvals = np.array([0., 0.25, 0.50, 0.75, 1.])
        yvals = np.array([0., 0.60, 0.85, 0.98, 1.])
        return xvals, yvals


    def getBetaCurve(self):
        """ Define default control points for Beta bezier control curve"""
        xvals = np.array([0., 0.25, 0.50, 0.75, 1.])
        yvals = np.array([0., 0.25, 0.50, 0.75, 1.])
        return xvals, yvals


    def gaussian_curvature(self,Z,dx,dy):
        Zy, Zx = np.gradient(Z,dx,dy)
        Zxy, Zxx = np.gradient(Zx)
        Zyy, _ = np.gradient(Zy)
        K = (Zxx * Zyy - (Zxy ** 2)) / (1 + (Zx ** 2) + (Zy ** 2)) ** 2
        return K


    def surfature(self, X, Y, Z):
        """ Compute Gaussian and mean curvatures of a surface K, H = surfature(X,Y,Z),
        where X,Y,Z are 2d arrays of points on the surface.  K and H are the Gaussian and mean curvatures, respectively.
        """
        # where X, Y, Z matrices have a shape (lr,lb)
        lr = X.shape[0]
        lb = X.shape[1]

        # First Derivatives
        Xv, Xu = np.gradient(X)
        Yv, Yu = np.gradient(Y)
        Zv, Zu = np.gradient(Z)

        # Second Derivatives
        Xuv, Xuu = np.gradient(Xu)
        Yuv, Yuu = np.gradient(Yu)
        Zuv, Zuu = np.gradient(Zu)

        Xvv, Xuv = np.gradient(Xv)
        Yvv, Yuv = np.gradient(Yv)
        Zvv, Zuv = np.gradient(Zv)

        # Reshape to 1D vectors
        nrow = lr * lb  # total number of rows after reshaping
        Xu = Xu.reshape(nrow, 1)
        Yu = Yu.reshape(nrow, 1)
        Zu = Zu.reshape(nrow, 1)
        Xv = Xv.reshape(nrow, 1)
        Yv = Yv.reshape(nrow, 1)
        Zv = Zv.reshape(nrow, 1)
        Xuu = Xuu.reshape(nrow, 1)
        Yuu = Yuu.reshape(nrow, 1)
        Zuu = Zuu.reshape(nrow, 1)
        Xuv = Xuv.reshape(nrow, 1)
        Yuv = Yuv.reshape(nrow, 1)
        Zuv = Zuv.reshape(nrow, 1)
        Xvv = Xvv.reshape(nrow, 1)
        Yvv = Yvv.reshape(nrow, 1)
        Zvv = Zvv.reshape(nrow, 1)

        Xu = np.c_[Xu, Yu, Zu]
        Xv = np.c_[Xv, Yv, Zv]
        Xuu = np.c_[Xuu, Yuu, Zuu]
        Xuv = np.c_[Xuv, Yuv, Zuv]
        Xvv = np.c_[Xvv, Yvv, Zvv]

        # % First fundamental Coeffecients of the surface (E,F,G)
        E = np.einsum('ij,ij->i', Xu, Xu)
        F = np.einsum('ij,ij->i', Xu, Xv)
        G = np.einsum('ij,ij->i', Xv, Xv)

        m = np.cross(Xu, Xv, axisa=1, axisb=1)
        p = np.sqrt(np.einsum('ij,ij->i', m, m))
        n = m / np.c_[p, p, p]

        # % Second fundamental Coeffecients of the surface (L,M,N)
        L = np.einsum('ij,ij->i', Xuu, n)
        M = np.einsum('ij,ij->i', Xuv, n)
        N = np.einsum('ij,ij->i', Xvv, n)

        # % Gaussian Curvature
        K = (L * N - M ** 2) / (E * G - L ** 2)
        K = K.reshape(lr, lb)

        # % Mean Curvature
        H = (E * N + G * L - 2 * F * M) / (2 * (E * G - F ** 2))
        H = H.reshape(lr, lb)

        return H, K