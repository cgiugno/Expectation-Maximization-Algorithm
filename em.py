import math
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.patches as mpatches
from regex import D
from scipy.stats import multivariate_normal
import numpy as np
import math as math
import random as rand

from sklearn import datasets


inputFiles = ["emData/A", "emData/B", "emData/C", "emData/Z"]


class EMG:
    def __init__(self, fileName, dim, kVal, dataSetName):
        # self.inputX = np.ndarray(shape=(1, dim), dtype=float)
        # self.inputY = np.ndarray(shape=(1,1), dtype=float)
        self.dim = dim

        print(self.dim)
        self.kVal = kVal

        self.dataSetName = dataSetName

        inputX = []

        pi = []
        # sigma = np.ndarray(shape=(kVal, dim, dim))
        # mu = []

        gamma = []

        mu = 2*(np.random.random((kVal, dim))-0.5)  # +/- 1
        sigma = np.random.random((kVal, dim, dim))  # (0,1)
        # avoid ill-conditioned covariance matrices
        for k in range(0, kVal):
            sigma[k] = sigma[k].dot(sigma[k].transpose())

        # sigmaMultiplier = abs(1 + rand.gauss(0, 1))
        for k in range(0, kVal):
            # mu.append([0]* dim)
            pi.append(1.0 / kVal)
            # sigmak = np.identity(dim, dtype=float)
            # for d in range(0, dim):
              # mu[k][d] = (1 + rand.gauss(0, 1))

            # for i in range(0, dim):
              # for d in range(0, dim):
                  # sigmak[i][d] *= sigmaMultiplier

            # sigma[k] = sigmak

        # Open input file
        with open(fileName, "r") as inputFile:
            # Read in file, line by line
            itr = 0
            for fileLine in inputFile:

                # Split the line using the character ','
                lineValues = fileLine.split(" ")
                # print("Line: \n", lineValues)

                # If there are at least two values on the line
                if len(lineValues) >= (dim):
                    inputX.append([0.0] * dim)
                    gamma.append([0.0] * kVal)

                    # if (itr < 100):
                      # print("Line %f, Length of X: %f, " %(itr, len(inputX)))
                    # Append the first to X value list and the second to the T value list
                    # print("Iteration: %d" %itr)
                    for i in range(0, dim):
                        inputX[itr][i] = (
                            float(lineValues[i].replace("\n", "")))
                        # print("Value: %s" %(inputX[itr][i]))
                    # print("\n")

                itr += 1

        # print("Gamma: \n", gamma)

        # Close the input file
        inputFile.close()

        xArray = np.ndarray(shape=(len(inputX), dim), dtype=float)
        piArray = np.ndarray(shape=(len(pi)), dtype=float)
        muArray = np.ndarray(shape=(len(mu), dim), dtype=float)
        gammaArray = np.ndarray(shape=(len(gamma), kVal), dtype=float)
        # print2DArray(inputX)
        # print("A %d by %d array" %(len(inputX), dim))
        for i in range(0, len(inputX)):
            for j in range(0, len(inputX[i])):
                # print("i: %f, j: %f\n" %(i, j))
                xArray[i][j] = inputX[i][j]
            for k in range(0, len(gamma[i])):
                gammaArray[i][k] = gamma[i][k]
        for k in range(0, len(mu)):
            piArray[k] = pi[k]
            for d in range(0, len(mu[k])):
                muArray[k][d] = mu[k][d]

        self.xMatrix = xArray
        self.piMatrix = piArray
        self.sigmaMatrix = sigma
        self.muMatrix = muArray
        self.gammaMatrix = gammaArray
        self.nk = np.ndarray(shape=(kVal), dtype=float)
        self.likelihoods = []

    # TESTED GAUSS, IT'S WORKING FINE
    def gaussian(self, xParams, muk, sigmak):

        # print("DATA POINT X: ", xParams)
        # print("MU VALUE: ", muk)
        # dim = float(len(xParams))
        # print("Dimensionality: ", dim, "\n")

        # term1 = (1.0 / (pow((2.0 * math.pi), (float(dim) / 2.0))))

        # mathfunc =  float(1) / float(6)
        # print("1/2pi^(D/2) = ", term1)
        # print(sigmak)

        # det = np.linalg.det(sigmak)

        # print("Determinant: ", det, "\n")

        # term2 = (1 / (np.linalg.det(sigmak) ** (1.0/2)))

        # print("1/|E|^(1/2) = ", term2)

        rv = multivariate_normal.pdf(xParams, muk, sigmak)
        # print("(x - mu)T: \n", np.transpose((xParams - muk)))
        # print("(x - mu): \n", (xParams - muk))
        # print("Inverse of Sigmak: \n",  np.linalg.inv(sigmak))
        # print("Matrix Multiplcation :\n", ((xParams - muk)) * np.linalg.inv(sigmak) * (xParams - muk))
        # print("Exponent: \n", (-1/2) * ((xParams - muk)) * np.linalg.inv(sigmak) * (xParams - muk))

        # term3 = np.exp((-1/2) * np.transpose((xParams - muk)) * np.linalg.inv(sigmak) * (xParams - muk))

        # print("exp((-1/2) * (x - mu)T * E-1 * (x - mu)) = ", (term3.item(0)))

        # print("Gauss to be returned: ", (term1 * term2 * term3.item(0)), "\n")

        return rv

    def origGauss(self, xParams, muk, sigmak):

        # print("DATA POINT X: ", xParams)
        # print("MU VALUE: ", muk)
        dim = float(len(xParams))
        # print("Dimensionality: ", dim, "\n")

        term1 = (1.0 / (pow((2.0 * math.pi), (float(dim) / 2.0))))

        # mathfunc =  float(1) / float(6)
        # print("1/2pi^(D/2) = ", term1)
        # print(sigmak)

        # det = np.linalg.det(sigmak)

        # print("Determinant: ", det, "\n")

        term2 = (1 / (np.linalg.det(sigmak) ** (1.0/2)))

        # print("1/|E|^(1/2) = ", term2)

        # rv = multivariate_normal.pdf(xParams, muk, sigmak)
        # print("(x - mu)T: \n", np.transpose((xParams - muk)))
        # print("(x - mu): \n", (xParams - muk))
        # print("Inverse of Sigmak: \n",  np.linalg.inv(sigmak))
        # print("Matrix Multiplcation :\n", ((xParams - muk)) * np.linalg.inv(sigmak) * (xParams - muk))
        # print("Exponent: \n", (-1/2) * ((xParams - muk)) * np.linalg.inv(sigmak) * (xParams - muk))

        term3 = np.exp((-1/2) * np.transpose((xParams - muk))
                       * np.linalg.inv(sigmak) * (xParams - muk))

        # print("exp((-1/2) * (x - mu)T * E-1 * (x - mu)) = ", (term3.item(0)))

        # print("Gauss to be returned: ", (term1 * term2 * term3.item(0)), "\n")

        return term1 * term2 * term3.item(0)

    def eStep(self):

        # print("Length: ", len(self.xMatrix[:,0]))

        for n in range(0, len(self.xMatrix[:, 0])):
            sumOfPiNormal = 0.0
            for j in range(0, len(self.muMatrix[:, 0])):
                # print("GAUSSIAN ITEMS: \nPi : ",  self.piMatrix[j])
                # print("\nX Matrix: ", self.xMatrix[n])
                # print("\nMu: ", self.muMatrix[j])
                # print("\nSigma: " + repr(self.sigmaMatrix[j]) + "\n")

                # print("n: %d\nk: %d\n" %(n, j))

                if (False):
                    print("Pi for 0: \n", self.piMatrix[j])
                    print("Mu for 0: \n", self.muMatrix[j])
                    print("X for 0: \n", self.xMatrix[j])
                    print("Sigma for 0: \n", self.sigmaMatrix[j])
                    print("Gauss for 0: \n", self.gaussian(
                        self.xMatrix[n], self.muMatrix[j], self.sigmaMatrix[j]))
                sumOfPiNormal += self.piMatrix[j] * self.gaussian(
                    self.xMatrix[n], self.muMatrix[j], self.sigmaMatrix[j])

            # print(sumOfPiNormal)

            sumOfPi = 0.0

            for k in range(0, len(self.muMatrix[:, 0])):
                # print("n: %d\nk: %d\n" %(n, k))
                individK = self.piMatrix[k] * self.gaussian(
                    self.xMatrix[n], self.muMatrix[k], self.sigmaMatrix[k])
                self.gammaMatrix[n][k] = individK / sumOfPiNormal

                sumOfPi += self.gammaMatrix[n][k]
                # print("Gamma (%d, %d): " %(n, k) + repr(self.gammaMatrix[n][k]) + "\n" )

            # print("Sum of Pi: ", sumOfPi)

            # print("Gamma Matrix: \n", self.gammaMatrix)

    def mStep(self, printStatements):
        nTotal = 0.0

        for k in range(0, self.kVal):
            self.nk[k] = 0.0

        for k in range(0, self.kVal):
            for n in range(0, len(self.gammaMatrix)):
                # print("K: %d" %k)
                # print("N: %d" %n)
                self.nk[k] += self.gammaMatrix[n][k]
                # print("NK[k]: ", self.nk[k])
            nTotal += self.nk[k]

        # print("Total N:", nTotal)

        # print("Gamma Matrix :\n", self.gammaMatrix)
        # print("Nk Matrix: \n", self.nk)

        for k in range(0, len(self.gammaMatrix[0, :])):
            # print("\nK: %d" %k)
            self.piMatrix[k] = self.nk[k] / nTotal
            # print("Pi for k", self.piMatrix[k])
            newMuk = 0.0
            for n in range(0, len(self.gammaMatrix)):
                # print("N: %d" %n)
                # print("Gamma for N, K: \n", self.gammaMatrix[n][k])
                # print("X for N: \n", self.xMatrix[n])
                newMuk += self.gammaMatrix[n][k] * self.xMatrix[n]
                # print("Current Muk: \n", newMuk)
            # print("Mu sum: ", newMuk)
            self.muMatrix[k] = (1.0 / self.nk[k]) * newMuk
            # print("New Mu: ", self.muMatrix[k])

        # print("Mu Matrix: \n", self.muMatrix)
        # print("Sigma Matrix: \n", self.sigmaMatrix)

        for k in range(0, len(self.gammaMatrix[0, :])):
            if printStatements:
                print("\nK VALUE: %d" % k)

            sigmakLength = len(self.sigmaMatrix[k])

            newSigmak = np.zeros(
                shape=(sigmakLength, sigmakLength), dtype=float)

            if printStatements:
                print("Sigma Matrix for k: \n", repr(self.sigmaMatrix[k]))
                print("New Sigma Matrix: \n", repr(newSigmak))

            for n in range(0, len(self.gammaMatrix)):
                if printStatements:
                    print("Data Point %d" % n)
                    print("Gamma :", self.gammaMatrix[n][k])
                    print("X Matrix: ", self.xMatrix[n])
                    print("Mu Matrix: ", self.muMatrix[k])
                xMinusMu = (self.xMatrix[n] - self.muMatrix[k])

                if printStatements:
                    print("x - mu: ", xMinusMu)
                xMinusMuDot = np.outer(xMinusMu, xMinusMu.T)

                if printStatements:
                    print("(x - mu) (x - mu)T\n", xMinusMuDot)
                    print("Final Value: \n",
                          self.gammaMatrix[n][k] * xMinusMuDot)

                newSigmak += self.gammaMatrix[n][k] * xMinusMuDot

                if printStatements:
                    print("Sigmak Afterward: \n", newSigmak)

            if printStatements:
                print(
                    "New Sigma after Summation, but before Normalization: \n", newSigmak)
                print("Multiply by 1/Nk = \n", (1.0 / self.nk[k]))
            self.sigmaMatrix[k] = (1.0 / self.nk[k]) * newSigmak
            # lazy technique for avoiding singularities
            self.sigmaMatrix[k] += 1e-3 * np.eye(self.dim)

            if printStatements:
                print("New Sigma: \n", self.sigmaMatrix[k], "\n")

    def EM(self, convergence, printStatements, itrToName, plotFig, saveFig):
        print("BEGINNING OF EM")
        print("Initial Pi: \n", self.piMatrix)
        print("Initial Mu: \n", self.muMatrix)
        print("Initial Sigma: \n", self.sigmaMatrix)

        # if plotFig:
            # self.plotSortedPoints(0, saveFig)
        err = np.inf
        itr = 0
        lastLnp = (-1) * np.inf

        lastPi = np.full(shape=(self.kVal), fill_value=np.inf)
        lastMu = np.full(shape=(self.kVal, self.dim), fill_value=np.inf)
        lastSigma = np.full(
            shape=(self.kVal, self.dim, self.dim), fill_value=np.inf)

        finalMu = np.inf
        finalPi = np.inf
        finalSigma = np.inf

        # print("Err: \n", err)
        # print("Convergence: \n", convergence)
        # print("Err < Convergence? \n", (err < convergence))

        # ((finalMu > convergence) & (finalPi > convergence) & (finalSigma > convergence)): # &  (itr < 3):
        while (itr < convergence):
            if (True):
                print("\nITERATION %f" % itr)
            self.eStep()

            if printStatements:
                print("Current Gamma: \n", self.gammaMatrix)

            self.mStep(printStatements)

            if printStatements:
                print("Current Pi: \n", self.piMatrix)
                print("Current Mu: \n", self.muMatrix)
                print("Current Sigma: \n", self.sigmaMatrix)

            lnp = 0.0
            for n in range(0, len(self.gammaMatrix)):

                if printStatements:
                    print("DATA POINT %d" % n)

                ln = 0.0
                for k in range(0, len(self.gammaMatrix[n])):
                    if printStatements:
                        print("GROUP K %d" % k)
                    g = self.gaussian(
                        self.xMatrix[n], self.muMatrix[k], self.sigmaMatrix[k])
                    if printStatements:
                        print("Pi: ", self.piMatrix[k])
                        print("Gaussian: ", g)
                        print("Pi * Gaussian: ", self.piMatrix[k] * g)
                    ln += self.piMatrix[k] * g
                    if printStatements:
                        print("Total Summation so Far: ", ln)

                ln = np.log(ln)

                if (printStatements):
                    print("LN of SUM for K: ", ln)
                lnp += ln

                if (printStatements):
                    print("LN  P so FAR: ", lnp)
            err = lnp - lastLnp

            if (printStatements):
                print("Difference between Likelihoods: ", repr(err))
                print("Difference greater than Convergence? %s" %
                      (err > convergence))
            itr += 1
            self.likelihoods.append(lnp)
            lastLnp = lnp

            # sumPi = 0.0
            # sumSigma = 0.0
            # sumMu = 0.0

            # differenceMu = lastMu - self.muMatrix
            # differenceSigma = lastSigma - self.sigmaMatrix
            # differencePi = lastPi - self.piMatrix

            # for k in range(0, self.kVal):
                # sumPi += (differencePi[k]) ** 2
                # for d in range(0, self.dim):
                    # sumMu += differenceMu[k][d] ** 2
                    # for i in range(0, self.dim):
                        # sumSigma += differenceSigma[k][d][i] ** 2

            # finalMu = sumMu ** (1.0/2.0)
            # finalSigma = sumSigma ** (1.0/2.0)
            # finalPi = sumPi ** (1.0 / 2.0)

            # lastMu = self.muMatrix
            # lastSigma = self.sigmaMatrix
            # lastPi = self.piMatrix

            if (False):
                print("Pi: \n", self.piMatrix)
                print("Mu: \n", self.muMatrix)
                print("Sigma: \n", self.sigmaMatrix)

                self.plotSortedPoints(itr=itr)
                self.plotLikelihoods(itr=itr)

            # sumOfMu = 0.0

            # for k in range(0, self.kVal):
                # currMu = self.muMatrix[k]
                # for i in range(k, self.kVal):
                  # differenceOfMu = currMu - self.muMatrix[i]

                  # for d in range(0, self.dim):
                      # sumOfMu += differenceOfMu[d] ** 2

                  # finalOfMu = sumOfMu ** (1.0/2.0)

                  # if finalOfMu < sepVal:
                      # self.muMatrix[i] = 2*(np.random.random((self.dim))-0.5)
                      # self.sigmaMatrix[i] = np.random.random((self.dim,self.dim))  # (0,1)
                      # avoid ill-conditioned covariance matrices
                      # self.sigmaMatrix[i] = self.sigmaMatrix[i].dot(self.sigmaMatrix[i].transpose())

        print("Pi: \n", self.piMatrix)
        print("Mu: \n", self.muMatrix)
        print("Sigma: \n", self.sigmaMatrix)

        if plotFig:
            self.plotSortedPoints(itrToName, saveFig)
        self.plotLikelihoods(itrToName, False)

        return self.likelihoods[len(self.likelihoods) - 1]

    def plotLikelihoods(self, itr, saveFig):
        plt.plot(range(0, len(self.likelihoods)), self.likelihoods, ".-")

        plt.xlabel("Iterations")
        plt.ylabel("Likelihood")

        if saveFig:
            plt.savefig('Dataset%s/%sLikelihood%d.png' %
                        (self.dataSetName, self.dataSetName, itr), bbox_inches='tight')

        plt.show()

    def plotSortedPoints(self, itr, saveFig):

        self.eStep()

        arrayOfXMatrix = np.array(self.xMatrix[:, 0])
        arrayOfYMatrix = np.array(self.xMatrix[:, 1])
        # print(arrayOfXMatrix.__repr__())

        minX = math.floor(np.min(arrayOfXMatrix))
        maxX = math.ceil(np.max(arrayOfXMatrix))

        minY = math.floor(np.min(arrayOfXMatrix))
        maxY = math.ceil(np.max(arrayOfYMatrix))

        arrayOfColors = [[0] * 3] * self.kVal

        # print(arrayOfColors)

        patches = []
        fig, ax = plt.subplots()

        for k in range(0, self.kVal):
            # print(k)
            r = rand.randint(0, 255)
            g = rand.randint(0, 255)
            b = rand.randint(0, 255)
            # arrayOfColors[k] = [r / 255.0, g / 255.0, b / 255.0, 1]

            h = (((360 / self.kVal) * k) / 360)
            s = (0.7)
            v = 1

            arrayOfColors[k] = clr.hsv_to_rgb([h, s, v]).tolist()
            arrayOfColors[k].append(1)

            plt.plot(self.muMatrix[k][0], self.muMatrix[k]
                     [1], "o", c=arrayOfColors[k])

            arrayOfColorsForK = []
            for i in range(0, 5):
                arrayOfColorsForK.append(
                    [arrayOfColors[k][0], arrayOfColors[k][1], arrayOfColors[k][2], i / 5.0])

            x, y = np.mgrid[minX:maxX:.01, minY:maxY:.01]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            rv = multivariate_normal(self.muMatrix[k], self.sigmaMatrix[k])
            plt.contour(x, y, rv.pdf(pos), colors=arrayOfColorsForK)

            patch = mpatches.Patch(
                color=arrayOfColors[k], label="Group %d" % k)
            patches.append(patch)

            # y = multivariate_normal.pdf(x, mean=self.muMatrix[k], cov=self.sigmaMatrix[k])
            # plt.plot(x, y)

        ax.legend(handles=patches)

        # plt.scatter(arrayOfXMatrix, self.yArray)

        for n in range(0, len(arrayOfXMatrix)):
            # print(self.gammaMatrix)
            # print("Data Point %d" %n)
            # print(self.gammaMatrix[n])
            kValForN = np.argmax(self.gammaMatrix[n])
            # print(kValForN)
            plt.plot(arrayOfXMatrix[n], arrayOfYMatrix[n], ".", c=(
                arrayOfColors[kValForN][0], arrayOfColors[kValForN][1], arrayOfColors[kValForN][2], arrayOfColors[kValForN][3]))
            # plt.text(arrayOfXMatrix[n], arrayOfYMatrix[n], "%d" %n, c=(arrayOfColors[kValForN][0], arrayOfColors[kValForN][1], arrayOfColors[kValForN][2], arrayOfColors[kValForN][3]))

        plt.xlabel("X Values")
        plt.ylabel("Y Values")

        # plt.title("Points in Dataset")

        plt.xlim(minX, maxX)
        plt.ylim(minY, maxY)

        if saveFig:
            plt.savefig('Dataset%s/%sIteration%d.png' %
                        (self.dataSetName, self.dataSetName, itr), bbox_inches='tight')

        plt.show()

    def plotAssociatedPoints(self):

        arrayOfXMatrix = np.array(self.xMatrix[:, 0])
        arrayOfYMatrix = np.array(self.xMatrix[:, 1])
        # print(arrayOfXMatrix.__repr__())

        minX = math.floor(np.min(arrayOfXMatrix))
        maxX = math.ceil(np.max(arrayOfXMatrix))

        minY = math.floor(np.min(arrayOfXMatrix))
        maxY = math.ceil(np.max(arrayOfYMatrix))

        plt.scatter(arrayOfXMatrix, arrayOfYMatrix)

        plt.xlabel("X Values")
        plt.ylabel("Y Values")

        plt.title("Points in Dataset %s" % self.dataSetName)

        plt.xlim(minX, maxX)
        plt.ylim(minY, maxY)

        plt.savefig("Points/%sPoints.png" %
                    self.dataSetName, bbox_inches='tight')

        plt.show()

    def outputAttrToFile(self, fileName):
        output = open(fileName, "w")

        output.write("%d\n" % self.dim)
        output.write("%d\n" % self.kVal)

        output.write("PI\n")
        for p in range(0, len(self.piMatrix)):
            output.write("%f " % self.piMatrix[p])

        output.write("\n")

        output.write("MU\n")
        for m in range(0, len(self.muMatrix)):
            for mm in range(0, len(self.muMatrix[m])):
                output.write("%f " % self.muMatrix[m][mm])
            output.write("\n")

        output.write("SIGMA\n")

        for s in range(0, len(self.sigmaMatrix)):
            for i in range(0, len(self.sigmaMatrix[s])):
                for j in range(0, len(self.sigmaMatrix[s][i])):
                    output.write("%f " % self.sigmaMatrix[s][i][j])
                output.write("\n")
            output.write("\n")

        output.write("END")

        output.close()


def readInMGFromFile(fileName):
    with open(fileName, "r") as inputFile:
        indexLine = inputFile.readline()
        indexLine = indexLine.replace("\s+", "")
        index = int(indexLine)

        dimLine = inputFile.readline()
        dimLine = dimLine.replace("\s+", "")
        dim = int(dimLine)

        kLine = inputFile.readline()
        kLine = kLine.replace("\s+", "")
        kVal = int(kLine)

        emg = EMG(inputFiles[index], dim, kVal, fileName)

        piLine = inputFile.readline().replace("\s+", "").replace("\n", "")
        # print(piLine)

        if (piLine != "PI"):
            print("ERROR: NO PI\n")
            return None

        piVals = inputFile.readline().split(" ")
        piVals[len(piVals) - 1] = piVals[len(piVals) - 1].replace("\n", "")
        for k in range(0, emg.kVal):
            emg.piMatrix[k] = float(piVals[k])

        muLine = inputFile.readline().replace("\s+", "").replace("\n", "")

        if (muLine != "MU"):
            print("ERROR: NO MU\n")
            return None

        for k in range(0, emg.kVal):
            muVals = inputFile.readline().split(" ")
            muVals[len(muVals) - 1] = muVals[len(muVals) - 1].replace("\n", "")
            for d in range(0, emg.dim):
                emg.muMatrix[k][d] = float(muVals[d])

        sigmaLine = inputFile.readline().replace("\s+", "").replace("\n", "")

        if (sigmaLine != "SIGMA"):
            print("ERROR: NO SIGMA\n")
            return None

        for k in range(0, emg.kVal):
            for i in range(0, emg.dim):
                sigmaVals = inputFile.readline().split(" ")
                sigmaVals[len(sigmaVals) - 1] = sigmaVals[len(sigmaVals) - 1].replace("\n", "")
                for j in range(0, emg.dim):
                    emg.sigmaMatrix[k][i][j] = float(sigmaVals[j])

            inputFile.readline()

        if (inputFile.readline().replace("\s+", "").replace("\n", "") != "END"):
            print("ERROR: NO END\n")
            return None

    inputFile.close()
    print("PI: \n", emg.piMatrix)
    print("MU: \n", emg.muMatrix)
    print("SIGMA: \n", emg.sigmaMatrix)
    return emg

def getIndividualLikelihood(emg):
    lnp = 0.0
    for n in range(0, len(emg.gammaMatrix)):

        ln = 0.0
        for k in range(0, len(emg.gammaMatrix[n])):
            g = emg.gaussian(emg.xMatrix[n], emg.muMatrix[k], emg.sigmaMatrix[k])
            ln += emg.piMatrix[k] * g

            # print(ln)
              
        ln = np.log(ln)

        lnp += ln

    return lnp


def getIndividualBIC(emg):
    emg.eStep()

    
    lnp = getIndividualLikelihood(emg)

    k = (emg.kVal - 1) + (emg.kVal) * (emg.dim) + \
    (emg.kVal) * (((emg.dim) * (emg.dim - 1)) / 2.0)
    BIC = ((-2 * lnp) + (math.log(len(emg.xMatrix)) * k))
    return BIC


def print2DArray(twoDarray):
    for i in range(0, len(twoDarray)):
        print("Line %d: " % i)
        for j in range(0, len(twoDarray[i])):
            print("%s " % twoDarray[i][j])
        print("\n")


def runEMforK(BIC, inputFileSet, fileNum, fileDim, fileName, k):
    try:
        emg = EMG(inputFileSet[fileNum], fileDim, k, fileName)
        lklhd = emg.EM(0.01, False, k, False, False)
        emg.outputAttrToFile("Output%s/output%s%d.txt" %
                             (emg.dataSetName, emg.dataSetName, k))
        print("%d" % len(emg.xMatrix))
        k = (emg.kVal - 1) + (emg.kVal) * (emg.dim) + \
        (emg.kVal) * (((emg.dim) * (emg.dim - 1)) / 2.0)
        BIC.append((-2 * lklhd) + (math.log(len(emg.xMatrix)) * k))
        return BIC
    except ValueError:
        return runEMforK(BIC, inputFileSet, fileNum, fileDim, fileName, k)

def dimensionalityForDataset(dataSet):
    if (dataSet == 1):
        return 2
    elif (dataSet == 2):
        return 2
    elif (dataSet == 3):
        return 2
    else:
        return 8

def nameForDataset(dataSet):
    if (dataSet == 1):
        return "A"
    elif (dataSet == 2):
        return "B"
    elif (dataSet == 3):
        return "C"
    else:
        return "Z"

def terminalForDataset(dataSet):
    print("WELCOME TO THE DATASET TESTER!")
    print("Please select a functionality: \n")
    print("1. Read in and graph the parameters for the best models of this dataset (enter 1)")
    print("2. Train a new model")

    inputChoice = input("Please enter 1 or 2: ")

    choice = int(inputChoice)

    if (choice == 1):
        emg = readInMGFromFile("Best%s.txt" %nameForDataset(dataSet))

        emg.eStep()
        
        if (dataSet != 4):
            emg.plotSortedPoints(0, False)
        print("Likelihood: %f\n" %getIndividualLikelihood(emg))

    if (choice == 2):
        dim = dimensionalityForDataset(dataSet)
        k = int(input("Please enter the number of clusters you wish to model: "))

        conv = int(input("Please enter the number of iterations you wish the EM algorithm to run: "))

        emg = EMG(inputFiles[dataSet - 1], dim, k, nameForDataset(dataSet))

        if (dataSet != 4):
            emg.EM(conv, False, 0, True, False)
        else:
            emg.EM(conv, False, 0, False, False)

def terminal():
    print("WELCOME TO THE TERMINAL!")
    print("Please select the dataset you wish to examine:")
    print("1. Dataset A (Enter 1)")
    print("2. Dataset B (Enter 2)")
    print("3. Dataset C (enter 3)")
    print("4. Dataset Z (enter 4")

    inputDataSet = input("Enter a number from 1 to 4: ")

    dataSet = int(inputDataSet)

    if (dataSet == 1):
        terminalForDataset(1)
    elif (dataSet == 2):
        terminalForDataset(2)
    elif (dataSet == 3):
        terminalForDataset(3)
    elif (dataSet == 4):
        terminalForDataset(4)
    else:
        print("ERROR: You did not choose a dataset.")

terminal()

# BIC = []

# for i in range(2, 11):
    # BIC = runEMforK(BIC, inputFiles, 3, 8, "Z", i)
    # print("BIC: \n", BIC)


# plt.plot(range(0 + 2, len(BIC) + 2), BIC, ".-")

# plt.xlabel("Iterations")
# plt.ylabel("Likelihood")

# plt.savefig('Output%s/%sLikelihood.png' % ("Z", "Z"), bbox_inches='tight')

# plt.show()


# emgA = readInMGFromFile("BestA.txt")

# bicA = getIndividualBIC(emgA)

# print("BIC: %f\n" %bicA)

# emgB = readInMGFromFile("BestB.txt")

# bicB = getIndividualBIC(emgB)

# print("BIC: %f\n" %bicB)

# emgC = readInMGFromFile("BestC.txt")

# bicC = getIndividualBIC(emgC)

# print("BIC: %f\n" %bicC)

# emgZ = readInMGFromFile("BestZ.txt")

# bicZ = getIndividualBIC(emgZ)

# print("BIC: %f\n" %bicZ)

# print("Pi: \n", emg.piMatrix[0])
# print("Mu: \n", emg.muMatrix[0])
# print("Sigma: \n", emg.sigmaMatrix[0])
# print("X Array: \n", emg.xMatrix[0])

# gauss = emg.gaussian(emg.xMatrix[0], emg.muMatrix[0], emg.sigmaMatrix[0])
# origGauss = emg.origGauss(emg.xMatrix[0], emg.muMatrix[0], emg.sigmaMatrix[0])

# print("Gauss: ", gauss)
# print("Original Gauss: ", origGauss)

# emg.plotAssociatedPoints()


# emg.plotSortedPoints()

# emg.plotLikelihoods()

# Value Error

# What am I actually doing now?
# The initialized newSigmak matrix in the M step is weirdly initializing to something in the original sigma matrix
