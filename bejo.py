import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import permutations

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes,
    MessageHandler, filters
)

# ====== CONFIG ======
TOKEN = "7951498360:AAFvCovBQivTjTWe53VO-Lsr7B7FsHXhGCI"  # Ganti token bot kamu
PASARAN_MAP = {
    "sgp": "https://raw.githubusercontent.com/widaditulus/4D/main/sgp.csv",
    "hk": "https://raw.githubusercontent.com/widaditulus/4D/main/hk.csv",
    "sdy": "https://raw.githubusercontent.com/widaditulus/4D/main/sydney.csv",
    "toto": "https://raw.githubusercontent.com/widaditulus/4D/main/toto.csv",
    "taiwan": "https://raw.githubusercontent.com/widaditulus/4D/main/taiwan.csv",
    "china": "https://raw.githubusercontent.com/widaditulus/4D/main/china.csv",
    "magnum": "https://raw.githubusercontent.com/widaditulus/4D/main/magnum.csv"
}
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO
)

POSITIONS = ['AS', 'CP', 'KP', 'EK']
MODEL_LIST = ['xgb', 'rf', 'nb', 'dt']
MODEL_PATH_TEMPLATE = "model_{model}_{pos}.pkl"
PASARAN_DEFAULT = "sgp"

# ====== DATA LOAD & PREPROCESS ======
def load_and_clean_data(url):
    try:
        df = pd.read_csv(url)
        # Pastikan ada kolom tanggal dan angka
        if 'tanggal' not in df.columns or 'angka' not in df.columns:
            # Jika format beda, coba sesuaikan di sini
            df.columns = ['tanggal', 'angka']
        df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
        df = df.dropna(subset=['tanggal'])
        df = df.dropna(subset=['angka'])
        # Format angka jadi string 4 digit (jika 4D) atau 5 digit
        df['angka_str'] = df['angka'].astype(str).str.zfill(4)
        # Filter yang valid (4 digit angka)
        df = df[df['angka_str'].str.match(r'^\d{4}$')]
        df = df.drop_duplicates(subset=['tanggal']).sort_values('tanggal')
        return df.reset_index(drop=True)
    except Exception as e:
        logging.error(f"Load data error: {e}")
        return pd.DataFrame()

def add_lag_features(df, n_lag=3):
    for pos_idx, pos in enumerate(POSITIONS):
        for lag in range(1, n_lag + 1):
            df[f'{pos}_lag{lag}'] = df['angka_str'].shift(lag).apply(
                lambda x: int(x[pos_idx]) if pd.notnull(x) else -1
            )
    df = df.dropna().reset_index(drop=True)
    return df

def extract_features_targets(df):
    features = []
    targets = {pos: [] for pos in POSITIONS}
    for _, row in df.iterrows():
        x = [
            row['tanggal'].dayofweek,
            row['tanggal'].isocalendar().week
        ]
        for pos in POSITIONS:
            for lag in range(1, 4):
                x.append(row[f'{pos}_lag{lag}'])
        features.append(x)
        for i, pos in enumerate(POSITIONS):
            targets[pos].append(int(row['angka_str'][i]))
    return np.array(features), targets

# ====== TRAINING ======
def train_all_models(df):
    X, y_map = extract_features_targets(df)
    if X.shape[0] < 10:
        logging.warning("Data terlalu sedikit untuk training.")
        return

    for pos in POSITIONS:
        y = np.array(y_map[pos])
        # Pastikan label mulai dari 0 (XGB harus 0-based)
        y -= y.min()

        if len(set(y)) < 2:
            logging.warning(f"Label untuk posisi {pos} kurang variatif.")
            continue

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except Exception as e:
            logging.error(f"Train-test split error pada posisi {pos}: {e}")
            continue

        for model_name in MODEL_LIST:
            if model_name == 'xgb':
                model = XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    objective='multi:softprob',
                    num_class=10,
                    verbosity=0,
                    random_state=42
                )
            elif model_name == 'rf':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_name == 'nb':
                model = GaussianNB()
            elif model_name == 'dt':
                model = DecisionTreeClassifier(random_state=42)

            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            logging.info(f"Model {model_name} Posisi {pos} - Akurasi test: {acc*100:.2f}%")
            joblib.dump(model, MODEL_PATH_TEMPLATE.format(model=model_name, pos=pos))

# ====== MODEL LOAD ======
def load_models():
    models = {pos: {} for pos in POSITIONS}
    for pos in POSITIONS:
        for model_name in MODEL_LIST:
            path = MODEL_PATH_TEMPLATE.format(model=model_name, pos=pos)
            if os.path.exists(path):
                models[pos][model_name] = joblib.load(path)
    return models

# ====== HYBRID PREDIKSI ======
def hybrid_predict(models, X):
    pred_digit = {}
    am_digits = {pos: [] for pos in POSITIONS}
    for pos in POSITIONS:
        prob_agg = np.zeros(10)
        total_models = 0
        for model_name in MODEL_LIST:
            model = models[pos].get(model_name)
            if model:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X)[0]
                else:
                    # Jika model tidak ada predict_proba (misal NB kadang)
                    pred = model.predict(X)[0]
                    prob = np.eye(10)[pred]
                prob_agg += prob
                total_models += 1
        if total_models > 0:
            prob_agg /= total_models
            top3 = prob_agg.argsort()[-3:][::-1].tolist()
            am_digits[pos] = top3
            pred_digit[pos] = top3[0]
        else:
            pred_digit[pos] = -1
            am_digits[pos] = [0, 0, 0]
    return pred_digit, am_digits

# ====== FORMAT OUTPUT ======
def generate_colok_naga_3d(am_dict):
    digits = [am_dict.get(pos, [0])[0] for pos in POSITIONS]
    combos = set()
    for combo in permutations(digits, 3):
        combos.add(''.join(str(d) for d in combo))
    return sorted(combos)[:3]

def format_output(cb, am, cn, posisi_detail):
    cb_str = str(cb)
    am_str = ', '.join(str(d) for d in am)
    cn_str = '\n'.join(', '.join(list(c)) for c in cn)
    pos_str = '\n'.join([f"{pos}: {', '.join(str(d) for d in posisi_detail.get(pos, [])[:3])}" for pos in POSITIONS])
    return f"CB: {cb_str}\nAM: {am_str}\n\nCN:\n{cn_str}\n\n{pos_str}"

# ====== HANDLER ======
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Halo! Kirim perintah /p [pasaran] [DDMMYY] untuk prediksi.\n"
        "Contoh: /p sgp 120725\n"
        "Untuk training model: /train [pasaran]\n"
        "Untuk input data manual: /input [pasaran] [tanggal DDMMYY] [angka 4 digit]\n"
    )

async def train_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    pasaran = args[0].lower() if args else PASARAN_DEFAULT
    if pasaran not in PASARAN_MAP:
        await update.message.reply_text(f"Pasaran '{pasaran}' tidak dikenal.")
        return

    await update.message.reply_text(f"Mulai training model untuk {pasaran.upper()}...")
    df = load_and_clean_data(PASARAN_MAP[pasaran])
    if df.empty:
        await update.message.reply_text(f"Data pasaran {pasaran.upper()} kosong, training dibatalkan.")
        return
    df = add_lag_features(df)
    train_all_models(df)
    await update.message.reply_text("Training selesai dan model sudah tersimpan.")

# Fungsi input data manual via bot
DATA_MANUAL = {}  # format: {pasaran: pd.DataFrame}

async def input_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) != 3:
        await update.message.reply_text("Format salah. Gunakan: /input [pasaran] [tanggal DDMMYY] [angka 4 digit]")
        return
    pasaran = args[0].lower()
    tanggal_str = args[1]
    angka = args[2]
    if pasaran not in PASARAN_MAP:
        await update.message.reply_text(f"Pasaran '{pasaran}' tidak dikenal.")
        return
    try:
        tanggal = datetime.strptime(tanggal_str, "%d%m%y")
    except:
        await update.message.reply_text("Format tanggal salah. Gunakan DDMMYY.")
        return
    if not angka.isdigit() or len(angka) != 4:
        await update.message.reply_text("Angka harus 4 digit angka.")
        return

    # Ambil data manual atau buat baru
    df_manual = DATA_MANUAL.get(pasaran)
    if df_manual is None:
        # Ambil data asli dari URL
        df_manual = load_and_clean_data(PASARAN_MAP[pasaran])

    # Update data manual dengan input baru
    new_row = pd.DataFrame({'tanggal': [tanggal], 'angka_str': [angka]})
    df_manual = pd.concat([df_manual, new_row], ignore_index=True)
    df_manual = df_manual.drop_duplicates(subset=['tanggal']).sort_values('tanggal').reset_index(drop=True)
    DATA_MANUAL[pasaran] = df_manual

    await update.message.reply_text(f"Data untuk {pasaran.upper()} tanggal {tanggal.strftime('%d-%m-%Y')} berhasil ditambahkan/diupdate.")

async def prepare_features_for_prediction(row):
    x = [
        row['tanggal'].dayofweek,
        row['tanggal'].isocalendar().week
    ]
    for pos_idx, pos in enumerate(POSITIONS):
        for lag in range(1, 4):
            x.append(int(row[f'{pos}_lag{lag}']) if f'{pos}_lag{lag}' in row else 0)
    return np.array(x).reshape(1, -1)

async def prediksi_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    pasaran = PASARAN_DEFAULT
    tanggal_str = None

    if len(args) >= 1:
        pasaran = args[0].lower()
    if len(args) == 2:
        tanggal_str = args[1]

    if pasaran not in PASARAN_MAP:
        await update.message.reply_text(f"Pasaran '{pasaran}' tidak dikenali.")
        return

    # Pakai data manual kalau ada, kalau tidak ambil dari URL
    df = DATA_MANUAL.get(pasaran)
    if df is None:
        df = load_and_clean_data(PASARAN_MAP[pasaran])

    if df.empty:
        await update.message.reply_text(f"Data pasaran {pasaran.upper()} kosong, silakan coba lagi nanti.")
        return

    if tanggal_str:
        try:
            tanggal = datetime.strptime(tanggal_str, "%d%m%y").date()
        except ValueError:
            await update.message.reply_text("Format tanggal salah. Gunakan DDMMYY, contoh: 120725.")
            return
    else:
        tanggal = df['tanggal'].max().date()

    if tanggal not in df['tanggal'].dt.date.values:
        await update.message.reply_text(f"Data untuk tanggal {tanggal.strftime('%d-%m-%Y')} tidak tersedia.")
        return

    df = add_lag_features(df)
    row = df[df['tanggal'].dt.date == tanggal].iloc[0]

    features = await prepare_features_for_prediction(row)
    models = load_models()
    pred_digit, am_digits = hybrid_predict(models, features)

    cb = ''.join(str(pred_digit[pos]) for pos in POSITIONS)
    am = [am_digits[pos][0] for pos in POSITIONS]
    cn = generate_colok_naga_3d(am_digits)
    output_text = format_output(cb, am, cn, am_digits)

    await update.message.reply_text(f"Prediksi {pasaran.upper()} untuk tanggal {tanggal.strftime('%d-%m-%Y')}:\n\n{output_text}")

# ====== MAIN ======
def main():
    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("train", train_handler))
    application.add_handler(CommandHandler("p", prediksi_handler))
    application.add_handler(CommandHandler("input", input_handler))

    logging.info("Bot started...")
    application.run_polling()

if __name__ == "__main__":
    main()
