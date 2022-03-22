import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib.pyplot as plt

class Hydro(object):
    def __init__(self, Nx, Ny, dx, Ibias, width, convectionFactor):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx

        self.psi = np.zeros((self.Nx, self.Ny))
        self.omega = np.zeros((self.Nx, self.Ny))
        self.Ibias = Ibias
        self.width = width
        self.convectionFactor = convectionFactor

        self.length = self.Nx * self.dx
        self.x0 = .5 * (self.length - self.width)
        self.x1 = self.x0 + self.width

    def _indexToCoordinate(self, index):
        return (index % self.Nx, index // self.Nx)

    def _coordinateToIndex(self, i, j):
        return i+j*self.Nx

    def _laplacianDirichelet(self, i, j):
        columns = []
        values = []
        pref = 1 / (self.dx ** 2)

        columns.append(self._coordinateToIndex(i, j))
        values.append(-4 * pref)

        if i != self.Nx - 1:
            columns.append(self._coordinateToIndex(i + 1, j))
            values.append(pref)
        if i != 0:
            columns.append(self._coordinateToIndex(i - 1, j))
            values.append(pref)

        if j != self.Ny - 1:
            columns.append(self._coordinateToIndex(i, j + 1))
            values.append(pref)
        if j != 0:
            columns.append(self._coordinateToIndex(i, j - 1))
            values.append(pref)

        return columns, values

    def _laplacianPsi(self, row):
        pref = 1 / (self.dx**2)
        i, j = self._indexToCoordinate(row)
        rhs = 0.0

        columns, values = self._laplacianDirichelet(i, j)
        rows = np.repeat(row, len(columns)).tolist()

        if i == 0:
            rhs += -1 * pref * self._psiOutOfBoundsLeft(i,j)
        if i == self.Nx - 1:
            rhs += -1 * pref * self._psiOutOfBoundsRight(i,j)
        if j == 0:
            rhs += -1 * pref * self._psiOutOfBoundsBottom(i,j)
        if j == self.Ny - 1:
            rhs += -1 * pref * self._psiOutOfBoundsTop(i,j)

        return rows, columns, values, rhs

    def _laplacianOmega(self, row):
        pref = 1 / (self.dx ** 2)
        i, j = self._indexToCoordinate(row)

        columns, values = self._laplacianDirichelet(i, j)
        rows = np.repeat(row, len(columns)).tolist()
        rhs = 0.0

        if i == 0:
            rhs += -1 * pref * self._omegaOutOfBoundsLeft(i, j)
        if i == self.Nx - 1:
            rhs += -1 * pref * self._omegaOutOfBoundsRight(i, j)
        if j == 0:
            rhs += -1 * pref * self._omegaOutOfBoundsBottom(i, j)
        if j == self.Ny - 1:
            rhs += -1 * pref * self._omegaOutOfBoundsTop(i, j)

        return rows, columns, values, rhs

    def _convectionOmega(self, row):
        pref = 1 / (2*self.dx)
        i, j = self._indexToCoordinate(row)

        columns = []
        values = []
        rhs = 0.0

        dpsidx  =self._dpsidx(i, j)
        dpsidy = self._dpsidy(i, j)

        # We write the term -A (dpsi/dy domega/dx - dpsi/dx domega/dy)
        # First domega/dx
        if i == 0:
            columns.append(self._coordinateToIndex(i+1,j))
            values.append(-1*self.convectionFactor*dpsidy*pref)
            rhs += -1 * self.convectionFactor * dpsidy * self._omegaOutOfBoundsLeft(i,j)
        elif i == self.Nx-1:
            columns.append(self._coordinateToIndex(i - 1, j))
            values.append(1 * self.convectionFactor * dpsidy * pref)
            rhs += 1 * self.convectionFactor * dpsidy * self._omegaOutOfBoundsRight(i,j)
        else:
            columns.append(self._coordinateToIndex(i + 1, j))
            values.append(-1 * self.convectionFactor * dpsidy * pref)
            columns.append(self._coordinateToIndex(i - 1, j))
            values.append( 1 * self.convectionFactor * dpsidy * pref)

        # Then domega/dx
        if j == 0:
            columns.append(self._coordinateToIndex(i, j + 1))
            values.append(1 * self.convectionFactor * dpsidx * pref)
            rhs +=  1 * self.convectionFactor * dpsidx * self._omegaOutOfBoundsBottom(i, j)
        elif j == self.Ny - 1:
            columns.append(self._coordinateToIndex(i, j - 1))
            values.append(-1 * self.convectionFactor * dpsidx * pref)
            rhs += -1 * self.convectionFactor * dpsidx * self._omegaOutOfBoundsTop(i, j)
        else:
            columns.append(self._coordinateToIndex(i, j + 1))
            values.append( 1 * self.convectionFactor * dpsidx * pref)
            columns.append(self._coordinateToIndex(i, j - 1))
            values.append(-1 * self.convectionFactor * dpsidx * pref)

        rows = np.repeat(row, len(columns)).tolist()
        return rows, columns, values, rhs

    def _dpsidx(self, i, j):
        pref = 1 / (2*self.dx)

        if i == 0:
            dpsidx = pref * (self.psi[i+1,j] - self._psiOutOfBoundsLeft(i,j))
        elif i == self.Nx - 1:
            dpsidx = pref * (self._psiOutOfBoundsRight(i,j) - self.psi[i-1,j])
        else:
            dpsidx = pref * (self.psi[i+1,j] - self.psi[i-1,j])

        return dpsidx

    def _dpsidy(self, i, j):
        pref = 1 / (2*self.dx)

        if j == 0:
            dpsidy = pref * (self.psi[i,j+1] - self._psiOutOfBoundsBottom(i,j))
        elif j == self.Ny-1:
            dpsidy = pref * (self._psiOutOfBoundsTop(i,j) - self.psi[i,j-1])
        else:
            dpsidy = pref * (self.psi[i,j+1] - self.psi[i,j-1])

        return dpsidy

    def _psiOutOfBoundsTop(self, i, j):
        psiabove = 0.0
        if self.x0 <= i * self.dx < self.x1:
            psiabove = (i * self.dx - self.x0) * self.Ibias
        elif i * self.dx >= self.x1:
            psiabove = self.width * self.Ibias
        return psiabove

    def _psiOutOfBoundsBottom(self, i, j):
        psibelow = 0.0
        if self.x0 <= i * self.dx < self.x1:
            psibelow = (i * self.dx - self.x0) * self.Ibias
        elif i * self.dx >= self.x1:
            psibelow = self.width * self.Ibias
        return psibelow

    def _psiOutOfBoundsLeft(self, i, j):
        return 0.0

    def _psiOutOfBoundsRight(self, i, j):
        return self.Ibias * self.width

    def _omegaOutOfBoundsTop(self, i, j):
        return 2 * (self._psiOutOfBoundsTop(i,j) - self.psi[i,j]) / (self.dx**2)

    def _omegaOutOfBoundsBottom(self, i, j):
        return 2 * (self._psiOutOfBoundsBottom(i, j) - self.psi[i, j]) / (self.dx ** 2)

    def _omegaOutOfBoundsLeft(self, i, j):
        return 2 * (self._psiOutOfBoundsLeft(i, j) - self.psi[i, j]) / (self.dx ** 2)

    def _omegaOutOfBoundsRight(self, i, j):
        return 2 * (self._psiOutOfBoundsRight(i, j) - self.psi[i, j]) / (self.dx ** 2)

    def _mixedTermRhs(self, row):
        i, j = self._indexToCoordinate(row)
        pref = 1/self.dx

        # y-derivatives
        if j == 0:
            dpsidy   = .5 * pref * (self.psi[i,j+1] - self._psiOutOfBoundsBottom(i,j))
            domegady = .5 * pref * (self.omega[i,j+1] - self._omegaOutOfBoundsBottom(i,j))
        elif j == self.Ny - 1:
            dpsidy = .5 * pref * (self._psiOutOfBoundsTop(i,j) - self.psi[i, j-1])
            domegady = .5 * pref * (self._omegaOutOfBoundsTop(i,j) - self.omega[i, j-1])
        else:
            dpsidy = .5 * pref * (self.psi[i, j+1] - self.psi[i, j-1])
            domegady = .5 * pref * (self.omega[i, j+1] - self.omega[i, j-1])

        # x-derivatives
        if i == 0:
            dpsidx   = .5 * pref * (self.psi[i+1,j] - self._psiOutOfBoundsLeft(i,j))
            domegadx = .5 * pref * (self.omega[i+1,j] - self._omegaOutOfBoundsLeft(i,j))
        elif i == self.Nx - 1:
            dpsidx = .5 * pref * (self._psiOutOfBoundsRight(i,j) - self.psi[i-1, j])
            domegadx = .5 * pref * (self._omegaOutOfBoundsRight(i,j) - self.omega[i-1, j])
        else:
            dpsidx = .5 * pref * (self.psi[i+1, j] - self.psi[i-1, j])
            domegadx = .5 * pref * (self.omega[i+1, j] - self.omega[i-1,j])

        rhs = self.convectionFactor * (dpsidy * domegadx - dpsidx*domegady)

        return rhs

    def _makeSparseMatrix(self, rows, columns, values):
        coomatrix = sparse.coo_matrix((values, (rows, columns)),
                                      shape=(self.Nx*self.Ny, self.Nx*self.Ny))
        matrix = coomatrix.tocsr()
        return matrix

    def _solveMatrix(self, rows, columns, values, rhs):
        matrix = self._makeSparseMatrix(rows, columns, values)
        sol = sparse.linalg.spsolve(matrix, rhs)

        solgrid = sol.reshape((self.Ny, self.Nx)).T
        return solgrid

    def _generatePsi(self):
        rows = []
        cols = []
        vals = []
        rhs = []

        for i in range(self.Nx * self.Ny):
            newRows, newCols, newVals, newRhs = self._laplacianPsi(i)
            rows += newRows
            cols += newCols
            vals += newVals

            k, l = self._indexToCoordinate(i)

            rhs.append(newRhs - self.omega[k, l])

        return rows, cols, vals, rhs

    def _solvePsi(self):
        rows, cols, vals, rhs = self._generatePsi()
        newPsi = self._solveMatrix(rows, cols, vals, rhs)

        matrix = self._makeSparseMatrix(rows, cols, vals)

        return newPsi

    def _generateAndSolveOmega(self):
        rows = []
        cols = []
        vals = []
        rhs = []

        for i in range(self.Nx * self.Ny):
            # Laplacian term:
            newRows, newCols, newVals, laplacianRhs = self._laplacianOmega(i)
            rows += newRows
            cols += newCols
            vals += newVals

            # Convection term:
            newRows, newCols, newVals, convectionRhs = self._convectionOmega(i)
            rows += newRows
            cols += newCols
            vals += newVals

            # Ohmic term:
            rows += [i]
            cols += [i]
            vals += [-1]

            rhs.append(laplacianRhs + convectionRhs)

        newOmega = self._solveMatrix(rows, cols, vals, rhs)
        return newOmega

    def updatePsiPicard(self):
        self.psi = self._solvePsi()

    def updatePsiAnderson(self, alpha):
        self.psi = alpha * self._solvePsi()  + (1-alpha) * self.psi

    def updateOmegaPicard(self):
        self.omega = self._generateAndSolveOmega()

    def updateOmegaAnderson(self, alpha):
        self.omega = alpha * self._generateAndSolveOmega() + (1-alpha) * self.omega

    def residualPsi(self):
        rows, cols, vals, rhs = self._generatePsi()
        matrix = self._makeSparseMatrix(rows, cols, vals)
        lapl = (matrix@self.psi.T.flatten()).reshape((self.Ny, self.Nx)).T

        rhs = np.array(rhs).reshape((self.Ny, self.Nx)).T

        return lapl - rhs

    def iterateToConvergence(self, maxres, meanres, alpha=.3, iterationsBetweenPlots=-1, iterationsBetweenSpeedup=5,
                             maxitr=1000):
        res = 1e20
        itr = 0

        while np.max(res) > maxres or np.mean(res) > meanres:
            if itr == 0:
                self.updatePsiPicard()
                res = self.residualPsi()
                resprev = np.copy(res)
            else:
                self.updatePsiAnderson(alpha)

            self.updateOmegaAnderson(alpha)

            if iterationsBetweenSpeedup > -1:
                if itr % iterationsBetweenSpeedup == 0 and itr > 0:
                    res = self.residualPsi()
                    if np.max(np.abs(res)) - np.max(np.abs(resprev)) < 0:
                        alphaprev = alpha
                        alpha += .05 * (1 - alpha)
                        print("Speeding up simulation from", round(alphaprev, 3), "to", round(alpha, 3))
                    resprev = np.copy(res)

                if (itr - 1) % 5 == 0 and itr > 1:
                    res = self.residualPsi()
                    if np.max(np.abs(res)) - np.max(np.abs(resprev)) > 0:
                        alphaprev = alpha
                        alpha = .8 * alpha
                        print("Slowing down simulation from", round(alphaprev, 3), "to", round(alpha, 3))

                    resprev = np.copy(res)

            if iterationsBetweenPlots > -1:
                if itr % iterationsBetweenPlots == 0:
                    res = self.residualPsi()
                    plt.imshow(self.psi.T, origin='lower', cmap='turbo')
                    plt.colorbar()
                    plt.show()

            itr += 1
            if itr > maxitr:
                raise RuntimeError("Max. amount of iterations exceeded.")

    @property
    def currentDensity(self):
        Jx = np.zeros_like(self.psi)
        Jy = np.zeros_like(self.psi)

        for i in range(self.Nx):
            for j in range(self.Ny):
                Jx[i,j] = self._dpsidy(i,j)
                Jy[i,j] =-self._dpsidx(i,j)

        return Jx, Jy
