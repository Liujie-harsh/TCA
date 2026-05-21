from model import TimeSeriesCNN
import data_pp
from utils import untargeted_tca, evaluate_attack   


if __name__ == "__main__":
    # 简单测试
    model = TimeSeriesCNN(input_dim=6, seq_len=30)
    file_path = "/home/lj/Documents/TCN/data/3/household_power_consumption.txt"
    window_size = 30
    train_ratio = 0.7
    val_ratio = 0.15
    (X_train, y_train, X_val, y_val, X_test, y_test, 
     scaler_X, scaler_Y, feature_cols, target_col) = data_pp.load_and_preprocess_data(file_path, window_size, train_ratio, val_ratio, test_ratio)
    
    X = X_train
    Y = y_train

    X_adv_untargeted = untargeted_tca(model, X, Y)
    print("Untargeted Attack Evaluation:", evaluate_attack(model, X, X_adv_untargeted, Y))
    rmse,xsim,sim = evaluate_attack(model, X, X_adv_untargeted, Y).values()
    print(f"RMSE: {rmse:.4f}, XSIM: {xsim:.4f}, SIM: {sim:.4f}")