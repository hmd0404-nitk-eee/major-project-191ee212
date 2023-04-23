from master_import import *
from DataNormaliser import DataNormaliser
from ELMRegressor import ELMRegressor

def testELM(incl_train_range=True):
    hidden_neurons = [i for i in range(2, 19)]
    lag = [i for i in range(2, 19)]
    step_forward = [0, 1, 3, 6, 18]

    best_models = {}
    for step in step_forward:
        models = []
        for i in range(len(lag)):
            if incl_train_range:
                config = {"split": 0.8, "lag": lag[i], "step_forward": step, "Hidden Neurons": hidden_neurons[i], "Random State": 212, "Train Range": [1000, 10000]}
            else:
                config = {"split": 0.8, "lag": lag[i], "step_forward": step, "Hidden Neurons": hidden_neurons[i], "Random State": 212}

            elm = ELM(config)
            model = elm.train_and_predict(print_metrics=False)
            models.append(model)

        models.sort(key=lambda x: x[-1], reverse=True)
        best_models[step] = models[0]

        print("Timestep: ", step, " -> MAE: %.4f, RMSE: %.4f, R2: %.4f" % (best_models[step][-3], best_models[step][-2], best_models[step][-1]))

    for step in step_forward:
        trd, trs, td, ts, ty_pred, lag, mae, rmse, r2 = best_models[step]
        ty_pred = ty_pred.reshape(1, -1)[0]

        x_start = len(trs) - 100
        x_end = 100

        plt.figure(figsize=(15, 5))
        plt.plot(trs[x_start:len(trd) - step], trd[x_start:len(trd) - step], 'b', label="History")
        plt.plot(ts[:x_end], td[:x_end], 'g', label="Actual")
        plt.plot(ts[:x_end], ty_pred[:x_end], 'r--', label="Forecast")
        plt.axvline(trs[len(trd) - step-1], color="k", linestyle="--")
        plt.axvline(ts[0], color="k", linestyle="--")
        plt.grid()
        plt.legend()
        plt.xlabel("Samples")
        plt.ylabel("Signal Value")
        plt.show()

        # print("Lengths: Actual -> ", len(td), " Forecast -> ", len(ty_pred))
        # print("Lengths Samples: History -> ", len(trs), " Forecast -> ", len(ts))
        # print("Actual: ", td)
        # print("Forecast: ", ty_pred)


def sliding_window(serie, lag=2, step_forward=1):
    M = len(serie) # Length of time series

    X = np.zeros((M-(lag+step_forward-1), lag)) # Input definition
    y = np.zeros((M-(lag+step_forward-1), 1)) # Target definition

    cont = 0
    posinput = lag + cont
    posout = posinput + step_forward

    i = 0
    while posout<=M:
        X[i, :] = serie[cont:posinput]
        y[i] = serie[posout-1]
        cont+=1
        posinput = lag+cont
        posout = posinput + step_forward
        i+=1

    return X, y

class ELM():
    def __init__(self, elm_config: dict) -> None:
        self.config = elm_config #split, lag, step_forward, Hidden Neurons, Random State, Train Range
        self.data = None
        self.plot_config = {
            "figsize": (12,6),
            "grid": False,
            "xlabel": "X Label (units)",
            "ylabel": "Y Label (units)",
            "xRange_low": None,
            "xRange_high": None
        }        

    def print_error_metrics(self, y, y_pred):
        mae = mean_absolute_error(y, y_pred)
        rmse = math.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred) * 100

        print("-"*50)
        print("MAE: %0.4f" % mae)
        print("RMSE: %0.4f" % rmse)
        print("R2: %0.4f" % r2)
        print("-"*50)

    def load_data(self, src="./input_data.csv", print_data=True):
        data = pd.read_csv(src, header=None).values
        self.data = data.reshape(1, -1)[0]

        if "Train Range" in self.config.keys():
            self.data = self.data[self.config["Train Range"][0]:self.config["Train Range"][1]]

        if print_data:
            print(self.data)
    
    def plot_input_data(self, config: dict):
        for item in config.items():
            self.plot_config[item[0]] = item[1]

        low = self.plot_config["xRange_low"] if self.plot_config["xRange_low"] != None else 0
        high = self.plot_config["xRange_high"] if self.plot_config["xRange_high"] != None else len(self.data)

        plt.figure(figsize=self.plot_config["figsize"])
        plt.plot(self.data[low:high], label="Input Data")
        plt.xlabel(self.plot_config["xlabel"])
        plt.ylabel(self.plot_config["ylabel"])
        
        if self.plot_config["xRange_low"] != None and self.plot_config["xRange_high"] != None:
            if high - low + 1 < 1000:
                step_size = 50
            else:
                step_size = 500
            plt.xticks(ticks=[i for i in range(0, high-low + 1, step_size)], labels=[i for i in range(low, high+1, step_size)])

        if self.plot_config["grid"]:
            plt.grid()
        plt.legend()
        plt.show()

    def get_training_testing_data(self, split=0.75):
        samples = [i for i in range(len(self.data))]

        train_size = int(len(self.data) * split)
        train = self.data[:train_size]
        train_samples = samples[:train_size]
        test = self.data[train_size:]
        test_samples = samples[train_size:]

        return train, train_samples, test, test_samples

    def train_and_predict(self, print_metrics=True):
        if type(self.data) != list:
            self.load_data(print_data=False)

        dn = DataNormaliser([-1, 1])
        trd, trs, td, ts = self.get_training_testing_data(self.config["split"])

        dn.fit(trd)
        trd_norm = dn.transform(trd)
        td_norm = dn.transform(td)

        lag = self.config["lag"]
        step_forward = self.config["step_forward"]

        trainX_norm, trainY_norm = sliding_window(trd_norm, lag, step_forward)
        testX_norm, testY_norm = sliding_window(td_norm, lag, step_forward)

        L = self.config["Hidden Neurons"]
        random_state = self.config["Random State"]

        elm = ELMRegressor(L, random_state)
        elm.fit(trainX_norm, trainY_norm)

        try_pred_norm = elm.predict(trainX_norm)
        ty_pred_norm = elm.predict(testX_norm)
        try_pred = dn.inverse_transform(try_pred_norm)
        ty_pred = dn.inverse_transform(ty_pred_norm)

        _, y_train = sliding_window(trd)
        _, y_test = sliding_window(td)

        if print_metrics:
            print("Training Errors:")
            size = min(len(y_train), len(try_pred))
            self.print_error_metrics(y_train[:size], try_pred[:size])
            print("\nTesting Errors:")
            size = min(len(y_test), len(ty_pred))
            self.print_error_metrics(y_test[:size], ty_pred[:size])
        else:
            size = min(len(y_test), len(ty_pred))
            mae = mean_absolute_error(y_test[:size], ty_pred[:size])
            rmse = math.sqrt(mean_squared_error(y_test[:size], ty_pred[:size]))
            r2 = r2_score(y_test[:size], ty_pred[:size]) * 100

            return [trd, trs, td, ts, ty_pred, lag, mae, rmse, r2]