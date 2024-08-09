import torch
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler,MinMaxScaler


np.random.seed(2333)
torch.set_default_tensor_type(torch.DoubleTensor)


def load_raw_data(data_path):
    # raw_data = np.loadtxt(data_path, delimiter=',')
    raw_data = np.array(pd.read_csv(data_path, header=None).values.astype(float))
    scaler = StandardScaler()
    Data = scaler.fit_transform(raw_data)
    return Data, scaler

def load_raw_data_min(data_path):
    # raw_data = np.loadtxt(data_path, delimiter=',')
    raw_data = np.array(pd.read_csv(data_path, header=None).values.astype(float))
    scaler = MinMaxScaler()
    Data = scaler.fit_transform(raw_data)
    return Data, scaler

def load_raw_data_min2(data_path):
    df = pd.read_csv(data_path, header=None)
    df = df.replace(0, np.nan)
    df = df.fillna(method='ffill')
    df = df.fillna(0.0)
    raw_data = np.array(df.values.astype(float))
    scaler = MinMaxScaler()
    Data = scaler.fit_transform(raw_data)
    return Data, scaler


def load_raw_data_2(file_path):  # Inverse normalized data's parameter
    Data2 = np.loadtxt(file_path, delimiter=",", skiprows=0)
    Data2 = torch.from_numpy(Data2.T)
    # print(Data2)
    Min_Val = Data2.min()
    value = Data2 - Data2.min()
    Sum = value.sum(dim=-1, keepdim=True).clamp_(min=1.)
    return Min_Val, Sum


def tuning_data(data_path):
    # raw_data = np.loadtxt(data_path, delimiter=',')
    df = pd.read_csv(data_path, header=None)
    df = df.interpolate(method='linear', limit_direction='forward')
    df = df.interpolate(method='linear', limit_direction='backward')
    raw_data = np.array(df.values.astype(float))
    scaler = StandardScaler()
    Data = scaler.fit_transform(raw_data)
    return Data, scaler


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1]), float(i[2])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j, z in edges:
        # z = z / 100  # construct adj matrix
        # z2 = z * z
        # A[i, j] = np.exp(-z2 / 10)
        # A[j, i] = np.exp(-z2 / 10)
        A[i, j] = 1
        A[j, i] = 1

    return A


def get_adj_bay_matrix(distance_df_filename, num_of_vertices):
    A = np.load(distance_df_filename)
    A = A - np.identity(num_of_vertices)
    return A

def get_adj_sz_matrix(distance_df_filename, num_of_vertices):
    A = np.array(pd.read_csv(distance_df_filename, header=None).values.astype(float))
    A = A - np.identity(num_of_vertices)
    return A

def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W


def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)


def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam-np.eye(n)


def get_mask_matrix(rate, arr):  # arr_size: the size of the arr
    zeros_nums = int(arr.size * rate)
    new_matrix = np.ones(arr.size)
    new_matrix[: zeros_nums] = 0
    np.random.shuffle(new_matrix)
    return new_matrix.reshape(arr.shape)


def get_data_matrix(raw_data_matrix, rate):
    np_data = np.array(raw_data_matrix)
    mask_matrix = get_mask_matrix(rate=rate, arr=np_data)
    data_matrix = np.multiply(raw_data_matrix, mask_matrix)
    return data_matrix, np.array(mask_matrix)


def load_data(len_train, len_val, len_test, rate, data_path):
    df, scaler = load_raw_data(data_path)
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val: len_train + len_val + len_test]

    train_data, train_m = get_data_matrix(train, rate)
    val_data, val_m = get_data_matrix(val, rate)
    test_data, test_m = get_data_matrix(test, rate)

    return train, val, test, train_data, val_data, test_data, train_m, val_m, test_m,scaler


def load_data_min2(len_train, len_val, len_test, rate, data_path):
    df, scaler= load_raw_data_min2(data_path)
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val: len_train + len_val + len_test]

    train_data, train_m = get_data_matrix(train, rate)
    val_data, val_m = get_data_matrix(val, rate)
    test_data, test_m = get_data_matrix(test, rate)

    return train, val, test, train_data, val_data, test_data, train_m, val_m, test_m, scaler

def load_data_min(len_train, len_val, len_test, rate, data_path):
    df, scaler= load_raw_data_min(data_path)
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val: len_train + len_val + len_test]

    train_data, train_m = get_data_matrix(train, rate)
    val_data, val_m = get_data_matrix(val, rate)
    test_data, test_m = get_data_matrix(test, rate)

    return train, val, test, train_data, val_data, test_data, train_m, val_m, test_m, scaler

def load_tuning_data(len_pretrain, len_train, len_val, len_test, data_path):
    df, scaler = tuning_data(data_path)
    train = df[len_pretrain: len_pretrain + len_train]
    val = df[len_pretrain + len_train: len_pretrain + len_train + len_val]
    test = df[len_pretrain + len_train + len_val:  len_pretrain + len_train + len_val + len_test]
    return train, val, test, scaler


def data_transform_prediction(data, n_his, n_pred, day_slot, device):
    n_day = len(data) // day_slot
    n_route = data.shape[1]
    n_slot = day_slot - n_his - n_pred + 1
    x = np.zeros([n_day * n_slot, 1, n_his, n_route])
    y = np.zeros([n_day * n_slot, n_pred, n_route])
    for i in range(n_day):
        for j in range(n_slot):
            t = i * n_slot + j
            s = i * day_slot + j
            e = s + n_his
            x[t, :, :, :] = data[s:e].reshape(1, n_his, n_route)
            y[t, :, :] = data[e:e + n_pred].reshape(n_pred, n_route)
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)


def data_transform_label(data, n_his, n_pred, day_slot, device):
    n_day = len(data) // day_slot
    n_route = data.shape[1]
    n_slot = day_slot - n_his - n_pred + 1
    x = np.zeros([n_day * n_slot, 1, n_his, n_route])
    for i in range(n_day):
        for j in range(n_slot):
            t = i * n_slot + j
            s = i * day_slot + j
            e = s + n_his
            x[t, :, :, :] = data[s:e].reshape(1, n_his, n_route)
    return torch.Tensor(x).to(device)


def data_transform_train(data, n_his, n_pred, day_slot, device):
    n_day = len(data) // day_slot
    n_route = data.shape[1]
    n_slot = day_slot - n_his - n_pred + 1
    x = np.zeros([n_day * n_slot, 1, n_his, n_route])
    for i in range(n_day):
        for j in range(n_slot):
            t = i * n_slot + j
            s = i * day_slot + j
            e = s + n_his
            x[t, :, :, :] = data[s:e].reshape(1, n_his, n_route)
    return torch.Tensor(x).to(device)


def mask_transform_min_batch(data, n_his, n_pred, day_slot, device):
    n_day = len(data) // day_slot
    n_route = data.shape[1]
    n_slot = day_slot - n_his - n_pred + 1
    x = np.zeros([n_day * n_slot, 1, n_his, n_route])
    for i in range(n_day):
        for j in range(n_slot):
            t = i * n_slot + j
            s = i * day_slot + j
            e = s + n_his
            x[t, :, :, :] = data[s:e].reshape(1, n_his, n_route)
    return torch.Tensor(x).to(device)


def random_transform_min_batch(data, n_his, n_pred, day_slot, device):
    n_day = len(data) // day_slot
    n_route = data.shape[1]
    n_slot = day_slot - n_his - n_pred + 1
    x = np.zeros([n_day * n_slot, 1, n_his, n_route])
    for i in range(n_day):
        for j in range(n_slot):
            t = i * n_slot + j
            s = i * day_slot + j
            e = s + n_his
            x[t, :, :, :] = data[s:e].reshape(1, n_his, n_route)
    return torch.Tensor(x).to(device)


def mse_test_loss(X, M, Label):
    MSE_test_loss = torch.mean(((1-M) * X - (1-M)*Label)**2) / torch.mean(1-M)
    return MSE_test_loss


def metric_test(X, M, Label):
    MAE_test_loss = torch.mean(torch.abs((1 - M) * X - (1 - M) * Label)) / torch.mean(1 - M)
    MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * Label) ** 2) / torch.mean(1 - M)
    index = torch.where((Label != 0) & (M == 0))
    MAPE_test_loss = torch.sum(torch.abs(Label[index]-X[index])/Label[index])/Label[index].shape[0]
    return MSE_test_loss, MAPE_test_loss * 100, MAE_test_loss


def mse_train_loss(X, M, Label):
    MSE_train_loss = torch.mean((M * X - M * Label) ** 2) / torch.mean(M)
    return MSE_train_loss


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    mse = masked_mse(pred, real, 0.0).item()
    return mae, mape, mse




def evaluate_prediction_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x)
            l_loss = loss(y_pred, y)
            l_sum += l_loss.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def evaluate_model(model, data_iter, loss):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            model_outputs = model(x)
            l_loss = loss(model_outputs, y)
            l_sum += l_loss.item()* y.shape[0]
            # print(l_sum)
            n +=  y.shape[0]
        return l_sum / n


def tuning_evaluate_model(model, data_iter, loss, n_his, node_nums, device):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter.get_iterator():
            x = torch.from_numpy(x[:, :, :, :]).to(device)
            y = torch.from_numpy(y[:, :, :, 0]).to(device)
            model_outputs = model(x.permute(0, 3, 1, 2))
            l_loss = loss(model_outputs, y)
            l_sum += l_loss.item()* y.shape[0]
            # print(l_sum)
            n += y.shape[0]
        return l_sum / n

def tuning_evaluate_model_fine(model, mmdnet, data_iter, loss, n_his, node_nums, device):
    model.eval()
    mmdnet.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter.get_iterator():
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y[:, :, :, 0]).to(device)
            x_tar = torch.permute(x, (0, 3, 1, 2))
            source_x = model(x_tar)
            source_x, task_x = mmdnet(source_x)
            model_outputs = model(x.permute(0, 3, 1, 2)+task_x)
            l_loss = loss(model_outputs, y)
            l_sum += l_loss.item()* y.shape[0]
            # print(l_sum)
            n +=  y.shape[0]
        return l_sum / n


def evaluate_model_2(model, data_iter, scaler, device):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            label_y = y[:, 0, :, :]
            mask = y[:, 1, :, :]
            y_pred = model(x)
            # y_pred = scaler.inverse_transform(y_pred.cpu())
            # y_pred = torch.Tensor(y_pred).to(device)
            # label_y = scaler.inverse_transform(label_y.cpu())
            # label_y = torch.Tensor(label_y).to(device)
            l_loss = mse_test_loss(X=y_pred, M=mask, Label=label_y)
            l_sum += l_loss.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def tuning_evaluate_metric(model, data_iter, scaler, n_his, logger, device):
    model.eval()
    with torch.no_grad():
        amae, amape, amse = [], [], []
        for i in range(12):
            labels = []
            preds = []
            for x, y in data_iter.get_iterator():
                x = torch.from_numpy(x[:, :, :, :]).to(device)
                y = torch.from_numpy(y[:, :, :, 0]).to(device)
                y_pred = model(x.permute(0, 3, 1, 2))
                y_pred_inverse = scaler.inverse_transform(y_pred[:, i, :].cpu())
                y_pred[:, i, :] = torch.Tensor(y_pred_inverse).to(device)
                y_inverse = scaler.inverse_transform(y[:, i, :].cpu())
                y[:, i, :] = torch.Tensor(y_inverse).to(device)
                labels.append(y[:, i, :])
                preds.append(y_pred[:, i, :])
            labels = torch.cat(labels, dim=0)
            preds = torch.cat(preds, dim=0)
            metrics = metric(preds, labels)
            MAE = metrics[0]
            MAPE = metrics[1]
            MSE = metrics[2]
            logger.info('Evaluate best model on test data for horizon {:d}\t MAE={:.6f}\t MAPE={:.6f}\t RMSE={:.6f}\t'
                        .format(i+1, MAE, MAPE*100, np.sqrt(MSE)))
            # print(log.format(i + 1, MAE, MAPE*100, RMSE))
            amae.append(MAE)
            amape.append(MAPE)
            amse.append(MSE)
        return np.array(amae).mean(), np.array(amape).mean()*100, np.sqrt(np.array(amse).mean())




def tuning_evaluate_metric_fine(model, mmdnet,data_iter, scaler, n_his, logger, node_nums, device):
    model.eval()
    mmdnet.eval()
    with torch.no_grad():
        amae, amape, amse = [], [], []
        for i in range(12):
            labels = []
            preds = []
            for x, y in data_iter.get_iterator():
                x = torch.from_numpy(x[:, :, :, :]).to(device)
                y = torch.from_numpy(y[:, :, :, 0]).to(device)
                x_tar = torch.permute(x, (0, 3, 1, 2))
                source_x = model(x_tar)
                source_x, task_x = mmdnet(source_x)
                y_pred = model(x.permute(0, 3, 1, 2)+task_x)
                y_pred_inverse = scaler.inverse_transform(y_pred[:, i, :].cpu())
                y_pred[:, i, :] = torch.Tensor(y_pred_inverse).to(device)
                y_inverse = scaler.inverse_transform(y[:, i, :].cpu())
                y[:, i, :] = torch.Tensor(y_inverse).to(device)
                labels.append(y[:, i, :])
                preds.append(y_pred[:, i, :])
            labels = torch.cat(labels, dim=0)
            preds = torch.cat(preds, dim=0)
            metrics = metric(preds, labels)
            MAE = metrics[0]
            MAPE = metrics[1]
            MSE = metrics[2]
            logger.info('Evaluate best model on test data for horizon {:d}\t MAE={:.6f}\t MAPE={:.6f}\t RMSE={:.6f}\t'
                        .format(i + 1, MAE, MAPE * 100, np.sqrt(MSE)))
            # print(log.format(i + 1, MAE, MAPE*100, RMSE))
            amae.append(MAE)
            amape.append(MAPE)
            amse.append(MSE)
        return np.array(amae).mean(), np.array(amape).mean() * 100, np.sqrt(np.array(amse).mean())


def evaluate_metric(model, data_iter, scaler, device):
    model.eval()
    with torch.no_grad():
        l_sum, mape_sum, mae_sum, n = 0.0, 0.0, 0.0, 0
        # ground_truth = []
        # y_pred_value = []
        # mask_value = []
        amae, amape, armse = [], [], []
        for i in range(12):
            mae, mape, rmse = [], [], []
            for x, y in data_iter:
                y_pred = model(x)
                y_pred_inverse = scaler.inverse_transform(y_pred.cpu())
                y_pred = torch.Tensor(y_pred_inverse).to(device)
                y_inverse = scaler.inverse_transform(y.cpu())
                y = torch.Tensor(y_inverse).to(device)
                # ground_truth.extend(label_y[:, -1, 300].tolist())
                # y_pred_value.extend(y_pred[:, -1, 300].tolist())
                # mask_value.extend(mask[:, -1, 300].tolist())
                # l_loss = mse_test_loss(y_pred, mask, label_y)
                mae.append(metric(y_pred[:, i, :], y[:, i, :])[0])
                mape.append(metric(y_pred[:, i, :], y[:, i, :])[1])
                rmse.append(metric(y_pred[:, i, :], y[:, i, :])[2])
            # pd_y_pred_value = pd.DataFrame(y_pred_value)
            # pd_ground_truth = pd.DataFrame(ground_truth)
            # pd_mask_value = pd.DataFrame(mask_value)
            # pd_y_pred_value.to_csv("./pretrain_model/pemsbay/prediction.csv",header=False,index=False)
            # pd_ground_truth.to_csv("./pretrain_model/pemsbay/ground_truth.csv",header=False,index=False)

            # pd_mask_value.to_csv("./pretrain_model/pemsbay/mask_value.csv", header=False, index=False)
            MAE = np.array(mae).mean()
            MAPE = np.array(mape).mean()
            RMSE = np.array(rmse).mean()
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i + 1, MAE, MAPE * 100, RMSE))
            amae.append(MAE)
            amape.append(MAPE)
            armse.append(RMSE)
        return np.array(amae).mean(), np.array(amape).mean() * 100, np.array(armse).mean()


def evaluate_metric_pemsbay(model, mmdnet,data_iter, scaler, n_his, logger, node_nums, device):
    model.eval()
    mmdnet.eval()
    with torch.no_grad():
        amae, amape, amse = [], [], []
        for i in range(12):
            labels = []
            preds = []
            for x, y in data_iter.get_iterator():
                x = torch.from_numpy(x[:, :, :, :]).to(device)
                y = torch.from_numpy(y[:, :, :, 0]).to(device)
                x_tar = torch.permute(x, (0, 3, 1, 2))
                source_x = model(x_tar)
                source_x, task_x = mmdnet(source_x)
                y_pred = model(x.permute(0, 3, 1, 2)+task_x)
                y_pred_inverse = scaler.inverse_transform(y_pred[:, i, :].cpu())
                y_pred[:, i, :] = torch.Tensor(y_pred_inverse).to(device)
                y_inverse = scaler.inverse_transform(y[:, i, :].cpu())
                y[:, i, :] = torch.Tensor(y_inverse).to(device)
                labels.append(y[:, i, :])
                # print(y[:, i, :])
                preds.append(y_pred[:, i, :])
            labels = torch.cat(labels, dim=0)
            preds = torch.cat(preds, dim=0)
            print(i)
            metrics = metric(preds, labels)
            MAE = metrics[0]
            MAPE = metrics[1]
            MSE = metrics[2]
            pd_y_pred_value = pd.DataFrame(preds.cpu())
            pd_ground_truth = pd.DataFrame(labels.cpu())
            # pd_y_pred_value.to_csv("./file/prediction_bay.csv",header=False,index=False)
            # pd_ground_truth.to_csv("./file/ground_truth_bay.csv",header=False,index=False)
            # print(log.format(i + 1, MAE, MAPE*100, RMSE))
            amae.append(MAE)
            amape.append(MAPE)
            amse.append(MSE)
        return np.array(amae).mean(), np.array(amape).mean() * 100, np.sqrt(np.array(amse).mean())


def evaluate_metric_metrla(model, mmdnet,data_iter, scaler, n_his, logger, node_nums, device):
    model.eval()
    mmdnet.eval()
    with torch.no_grad():
        amae, amape, amse = [], [], []
        for i in range(12):
            labels = []
            preds = []
            for x, y in data_iter.get_iterator():
                x = torch.from_numpy(x[:, :, :, :]).to(device)
                y = torch.from_numpy(y[:, :, :, 0]).to(device)
                x_tar = torch.permute(x, (0, 3, 1, 2))
                source_x = model(x_tar)
                source_x, task_x = mmdnet(source_x)
                y_pred = model(x.permute(0, 3, 1, 2)+task_x)
                y_pred_inverse = scaler.inverse_transform(y_pred[:, i, :].cpu())
                y_pred[:, i, :] = torch.Tensor(y_pred_inverse).to(device)
                y_inverse = scaler.inverse_transform(y[:, i, :].cpu())
                y[:, i, :] = torch.Tensor(y_inverse).to(device)

                labels.append(y[:, i, :])
                preds.append(y_pred[:, i, :])
            labels = torch.cat(labels, dim=0)
            preds = torch.cat(preds, dim=0)
            metrics = metric(preds, labels)
            MAE = metrics[0]
            MAPE = metrics[1]
            MSE = metrics[2]
            print(i)
            pd_y_pred_value = pd.DataFrame(preds.cpu())
            pd_ground_truth = pd.DataFrame(labels.cpu())
            pd_y_pred_value.to_csv("./file/prediction_la.csv",header=False,index=False)
            pd_ground_truth.to_csv("./file/ground_truth_la.csv",header=False,index=False)
            # print(log.format(i + 1, MAE, MAPE*100, RMSE))
            amae.append(MAE)
            amape.append(MAPE)
            amse.append(MSE)
        return np.array(amae).mean(), np.array(amape).mean() * 100, np.sqrt(np.array(amse).mean())