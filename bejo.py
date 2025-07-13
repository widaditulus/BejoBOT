import os
import re
import csv
import logging
import requests
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
from itertools import permutations
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
TOKEN = "7951498360:AAFvCovBQivTjTWe53VO-Lsr7B7FsHXhGCI"
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
PRED_LOG_FILE = os.path.join(LOG_DIR, "prediction_log.csv")
EVAL_LOG_FILE = os.path.join(LOG_DIR, "evaluation_log.csv")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

POSITIONS = ['AS', 'CP', 'KP', 'EK']
PASARAN_DEFAULT = 'sgp'
MODEL_PATH_TEMPLATE = "model_{model}_{pos}.pkl"
MODEL_LIST = ['xgb', 'rf', 'nb', 'dt']

# ====== DATA LOADING ======
def load_data(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = ['tanggal', 'angka']
        df['tanggal'] = pd.to_datetime(df['tanggal'])
        df = df.tail(500)
        df['angka_str'] = df['angka'].astype(str).str.zfill(5)  # 5 digit angka main
        return df
    except Exception as e:
        logging.error(f"Load data error: {e}")
        return pd.DataFrame()

# ====== FEATURE EXTRACTION ======
def extract_features(df):
    df['hari'] = df['tanggal'].dt.dayofweek
    df['minggu'] = df['tanggal'].dt.isocalendar().week
    X = []
    y_map = {pos: [] for pos in POSITIONS}
    for _, row in df.iterrows():
        x_row = [row['hari'], row['minggu']]
        for d in row['angka_str']:
            x_row.append(int(d))
        X.append(x_row)
        for i, pos in enumerate(POSITIONS):
            y_map[pos].append(int(row['angka_str'][i]))
    return np.array(X), y_map

# ====== TRAIN MODEL ======
def train_all_models(df):
    X, y_map = extract_features(df)
    for pos in POSITIONS:
        y = np.array(y_map[pos])
        for model_name in MODEL_LIST:
            model = None
            if model_name == 'xgb':
                model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            elif model_name == 'rf':
                model = RandomForestClassifier()
            elif model_name == 'nb':
                model = GaussianNB()
            elif model_name == 'dt':
                model = DecisionTreeClassifier()
            model.fit(X, y)
            joblib.dump(model, MODEL_PATH_TEMPLATE.format(model=model_name, pos=pos))

# ====== LOAD MODELS ======
def load_models():
    models = {pos: {} for pos in POSITIONS}
    for pos in POSITIONS:
        for model_name in MODEL_LIST:
            path = MODEL_PATH_TEMPLATE.format(model=model_name, pos=pos)
            if os.path.exists(path):
                models[pos][model_name] = joblib.load(path)
    return models

# ====== HYBRID PREDICT ======
def hybrid_predict(models, X):
    pred_digit = {}
    am_digits = {pos: [] for pos in POSITIONS}
    for pos in POSITIONS:
        votes = []
        for model_name in MODEL_LIST:
            model = models[pos].get(model_name)
            if model:
                pred = model.predict(X)[0]
                votes.append(pred)
        if votes:
            # Voting sekaligus dapat 3 top digit terbanyak (frekuensi suara)
            from collections import Counter
            c = Counter(votes)
            top3 = [d for d, _ in c.most_common(3)]
            am_digits[pos] = top3
            pred_digit[pos] = top3[0]
        else:
            pred_digit[pos] = -1
            am_digits[pos] = []
    return pred_digit, am_digits

# ====== COLOK NAGA 3D & ANGKA MAIN 5 DIGIT ======
def generate_colok_naga_3d(hasil):
    # kombinasi 3 digit dari hasil posisi (4 posisi) => ambil 3 digit teratas tiap posisi dan generate kombinasi 3 digit
    candidates = []
    for combo in permutations([hasil['AS'], hasil['CP'], hasil['KP'], hasil['EK']], 3):
        # combo contoh: (AS, CP, KP)
        candidates.append(''.join(str(d) for d in combo))
    # Hapus duplikat dan urutkan
    candidates = sorted(set(candidates))
    return candidates[:5]  # Maksimal 5 digit colok naga

# ====== LOGGING ======
def simpan_log(pasaran, cb, am, cn, actual=''):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(PRED_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([now, pasaran, cb, '|'.join(str(d) for d in am['AS']),
                         '|'.join(str(d) for d in am['CP']),
                         '|'.join(str(d) for d in am['KP']),
                         '|'.join(str(d) for d in am['EK']),
                         ','.join(cn),
                         actual])

def load_feedback():
    if not os.path.exists(PRED_LOG_FILE):
        return pd.DataFrame()
    df = pd.read_csv(PRED_LOG_FILE, header=None,
                     names=['timestamp', 'pasaran', 'cb', 'am_AS', 'am_CP', 'am_KP', 'am_EK', 'cn', 'actual'],
                     dtype=str, na_filter=False)
    return df

# ====== HITUNG AKURASI ======
def hitung_akurasi(prediksi, actual):
    if not prediksi or not actual or len(prediksi) != len(actual):
        return 0.0
    benar = sum(p == a for p, a in zip(prediksi, actual))
    return round(benar / len(prediksi) * 100, 2)

# ====== LAPORAN AKURASI MINGGUAN ======
def laporan_akurasi_mingguan(pasaran):
    if not os.path.exists(EVAL_LOG_FILE):
        return f"Tidak ada data evaluasi untuk pasaran {pasaran.upper()}."
    df = pd.read_csv(EVAL_LOG_FILE, header=None,
                     names=['timestamp', 'pasaran', 'tanggal_prediksi', 'prediksi', 'actual', 'akurasi'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    seminggu_lalu = datetime.now() - timedelta(days=7)
    df = df[(df['pasaran'] == pasaran) & (df['timestamp'] >= seminggu_lalu)]
    if df.empty:
        return f"Tidak ada data evaluasi minggu ini untuk pasaran {pasaran.upper()}."
    rata2 = df['akurasi'].astype(float).mean()
    hasil_baik = df[df['akurasi'] >= 75].shape[0]
    total = df.shape[0]
    return (f"Laporan akurasi minggu ini untuk {pasaran.upper()} ({total} prediksi):\n"
            f"Rata-rata akurasi digit: {rata2:.2f}%\n"
            f"Prediksi dengan akurasi ≥ 75%: {hasil_baik}")

# ====== RETRAIN DENGAN FEEDBACK ======
def retrain_with_feedback(pasaran):
    df = load_data(PASARAN_MAP[pasaran])
    if df.empty:
        return None
    feedback_df = load_feedback()
    if not feedback_df.empty:
        for _, row in feedback_df.iterrows():
            try:
                dt = pd.to_datetime(row['timestamp']).date()
                angka = row['actual']
                if len(angka) == 5 and angka.isdigit():
                    if dt not in df['tanggal'].dt.date.values:
                        df = df.append({'tanggal': dt, 'angka': int(angka), 'angka_str': angka.zfill(5)}, ignore_index=True)
            except Exception as e:
                logging.error(f"Error saat update feedback ke data training: {e}")
    df.drop_duplicates(subset=['tanggal'], keep='last', inplace=True)
    df.sort_values(by='tanggal', inplace=True)
    df.reset_index(drop=True, inplace=True)
    train_all_models(df)
    return load_models()

# ====== COMMAND HANDLERS ======

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔥 Bot Prediksi 4D Hybrid 🔥\n"
        "Perintah:\n"
        "/predict [pasaran] - Prediksi 5 digit lengkap\n"
        "/p [pasaran] [YYMMDD] - Prediksi dengan tanggal\n"
        "/input [pasaran] [YYMMDD] [angka] - Input hasil undian 5 digit\n"
        "/train [pasaran] - Latih ulang model\n"
        "/feedback [pasaran] - Tampilkan akurasi minggu ini\n"
        "/help - Bantuan"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Perintah:\n"
        "/predict [pasaran] - Prediksi 5 digit\n"
        "/p [pasaran] [YYMMDD] - Prediksi dengan tanggal\n"
        "/input [pasaran] [YYMMDD] [angka] - Input hasil undian 5 digit\n"
        "/train [pasaran] - Latih ulang model\n"
        "/feedback [pasaran] - Tampilkan akurasi minggu ini\n"
        "Pasaran tersedia: " + ", ".join(PASARAN_MAP.keys())
    )

async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pasaran = PASARAN_DEFAULT
    if context.args:
        p = context.args[0].lower()
        if p in PASARAN_MAP:
            pasaran = p
        else:
            await update.message.reply_text(f"Pasaran '{p}' tidak dikenal, gunakan default {pasaran}.")
    await update.message.reply_text(f"Training ulang semua posisi untuk {pasaran.upper()}... tunggu ya...")
    train_all_models(load_data(PASARAN_MAP[pasaran]))
    await update.message.reply_text("Training selesai, semua model tersimpan.")

async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pasaran = PASARAN_DEFAULT
    tanggal = None
    if context.args:
        p = context.args[0].lower()
        if p in PASARAN_MAP:
            pasaran = p
        else:
            await update.message.reply_text(f"Pasaran '{p}' tidak dikenal, gunakan default {pasaran}.")
            return
        if len(context.args) > 1:
            try:
                tanggal = datetime.strptime(context.args[1], "%y%m%d").date()
            except:
                await update.message.reply_text("Format tanggal salah. Gunakan YYMMDD.")
                return
    df = load_data(PASARAN_MAP[pasaran])
    if df.empty:
        await update.message.reply_text("Gagal memuat data.")
        return
    models = load_models()
    if not models:
        await update.message.reply_text("Model belum dilatih. Gunakan /train terlebih dahulu.")
        return
    X_all, _ = extract_features(df)
    X_input = X_all[-1].reshape(1, -1)
    hasil, am = hybrid_predict(models, X_input)
    hasil_str = ''.join(str(hasil[pos]) for pos in POSITIONS)
    # AM gabungkan jadi string format 5 digit total
    am_list = []
    for pos in POSITIONS:
        am_list.extend([str(d) for d in am[pos]])
    am_str = ', '.join(am_list)
    cn_list = generate_colok_naga_3d(hasil)
    # Format CN seperti permintaan (3 digit per baris, maksimal 5 baris)
    cn_formatted = ''
    for i, cn in enumerate(cn_list):
        cn_formatted += ', '.join(list(cn)) + ',\n'
        if i >= 4:
            break
    msg = (
        f"CB: {hasil_str[0]}\n"
        f"AM: {am_str}\n\n"
        "CN:\n"
        f"{cn_formatted}\n"
    )
    # Detail posisi
    for pos in POSITIONS:
        msg += f"{pos}: {', '.join(str(d) for d in am[pos])}\n"
    await update.message.reply_text(msg)
    simpan_log(pasaran, hasil_str[0], am, cn_list)

async def p_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Alias /p sama dengan /predict
    await predict_command(update, context)

async def input_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Format: /input [pasaran] [YYMMDD] [angka 5 digit]
    if len(context.args) != 3:
        await update.message.reply_text(
            "Format salah! Gunakan:\n"
            "/input [pasaran] [YYMMDD] [angka 5 digit]\n"
            "Contoh: /input sgp 130725 45678"
        )
        return
    pasaran = context.args[0].lower()
    if pasaran not in PASARAN_MAP:
        await update.message.reply_text(f"Pasaran '{pasaran}' tidak dikenal.")
        return
    tanggal_str = context.args[1]
    angka = context.args[2]
    if not re.match(r'^\d{6}$', tanggal_str):
        await update.message.reply_text("Tanggal harus 6 digit format YYMMDD.")
        return
    if not re.match(r'^\d{5}$', angka):
        await update.message.reply_text("Angka harus 5 digit.")
        return
    try:
        tanggal = datetime.strptime(tanggal_str, "%y%m%d").date()
    except Exception:
        await update.message.reply_text("Format tanggal salah. Gunakan YYMMDD.")
        return
    angka = angka.zfill(5)
    # Simpan hasil aktual ke log prediksi (kolom actual)
    try:
        with open(PRED_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([now, pasaran, '', '', '', '', '', '', '', angka])
        await update.message.reply_text(f"Hasil undian {pasaran.upper()} tanggal {tanggal_str} = {angka} berhasil disimpan.\nModel akan retrain otomatis sekarang.")
    except Exception as e:
        await update.message.reply_text(f"Gagal menyimpan hasil: {e}")
        return
    # Auto-retrain
    models = retrain_with_feedback(pasaran)
    if models:
        await update.message.reply_text("Retrain model selesai setelah input data baru.")
    else:
        await update.message.reply_text("Retrain gagal, cek log.")

async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(f"Gunakan: /feedback [pasaran]\nContoh: /feedback hk")
        return

    pasaran = context.args[0].lower()
    if pasaran not in PASARAN_MAP:
        await update.message.reply_text(f"Pasaran '{pasaran}' tidak dikenal. Pilih dari: {', '.join(PASARAN_MAP.keys())}")
        return

    if not os.path.exists(EVAL_LOG_FILE):
        await update.message.reply_text(f"Tidak ada data evaluasi untuk pasaran {pasaran.upper()}.")
        return

    df = pd.read_csv(EVAL_LOG_FILE, header=None,
                     names=['timestamp', 'pasaran', 'tanggal_prediksi', 'prediksi', 'actual', 'akurasi'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    seminggu_lalu = datetime.now() - timedelta(days=7)
    df = df[(df['pasaran'] == pasaran) & (df['timestamp'] >= seminggu_lalu)]
    if df.empty:
        await update.message.reply_text(f"Tidak ada data evaluasi minggu ini untuk pasaran {pasaran.upper()}.")
        return

    rata2 = df['akurasi'].astype(float).mean()
    hasil_baik = df[df['akurasi'] >= 75].shape[0]
    total = df.shape[0]
    await update.message.reply_text(
        f"Laporan akurasi minggu ini untuk {pasaran.upper()} ({total} prediksi):\n"
        f"Rata-rata akurasi digit: {rata2:.2f}%\n"
        f"Prediksi dengan akurasi ≥ 75%: {hasil_baik}"
    )

async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Perintah tidak dikenal.\n"
        "Gunakan salah satu perintah berikut:\n"
        "/predict sgp\n"
        "/p sgp 250713\n"
        "/input sgp 130725 45678\n"
        "/train sgp\n"
        "/feedback sgp\n"
    )

def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("train", train_command))
    app.add_handler(CommandHandler("predict", predict_command))
    app.add_handler(CommandHandler("p", p_command))
    app.add_handler(CommandHandler("input", input_command))
    app.add_handler(CommandHandler("feedback", feedback_command))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    logging.info("Bot started...")
    app.run_polling()

if __name__ == "__main__":
    main()

