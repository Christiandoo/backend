import pandas as pd
from google_play_scraper import reviews, Sort, app

def scrape_with_fallbacks():
    print("==================================================")
    print("[1] Memeriksa koneksi ke Google Play Store...")
    print("==================================================")
    try:
        app_info = app('id.sevima.edlink', lang='id', country='id')
        print(f"[✓] Terhubung! Aplikasi: {app_info.get('title')}")
        print(f"[✓] Total ulasan di Play Store: {app_info.get('reviews'):,} ulasan\n")
    except Exception as e:
        print(f"[x] Gagal terhubung ke Play Store: {e}")
        print("    -> Saran: Periksa koneksi internet, matikan VPN jika aktif.\n")
        return

    print("==================================================")
    print("[2] Memulai Scraping Ulasan (Multi-Opsi)...")
    print("==================================================")
    
    results = []

    # Opsi 1: Tanpa Sort (Default Play Store)
    print("[+] Opsi 1: Scraping dengan konfigurasi standar (lang='id')...")
    try:
        results, _ = reviews(
            'id.sevima.edlink',
            lang='id',
            country='id',
            count=3000
        )
        print(f"    -> Hasil Opsi 1: {len(results)} ulasan")
    except Exception as e:
        print(f"    -> Opsi 1 Gagal: {e}")

    # Opsi 2: Jika Opsi 1 kosong, coba dengan Sort.MOST_RELEVANT
    if not results:
        print("\n[+] Opsi 2: Menggunakan Sort.MOST_RELEVANT...")
        try:
            results, _ = reviews(
                'id.sevima.edlink',
                lang='id',
                country='id',
                sort=Sort.MOST_RELEVANT,
                count=3000
            )
            print(f"    -> Hasil Opsi 2: {len(results)} ulasan")
        except Exception as e:
            print(f"    -> Opsi 2 Gagal: {e}")

    # Opsi 3: Jika masih kosong, hapus filter negara/bahasa
    if not results:
        print("\n[+] Opsi 3: Mengambil ulasan tanpa filter bahasa/negara...")
        try:
            results, _ = reviews(
                'id.sevima.edlink',
                count=3000
            )
            print(f"    -> Hasil Opsi 3: {len(results)} ulasan")
        except Exception as e:
            print(f"    -> Opsi 3 Gagal: {e}")

    # Jika semua opsi tetap 0
    if not results:
        print("\n[x] Seluruh opsi menghasilkan 0 data.")
        print("    -> Pastikan Anda sudah menjalankan 'pip install --upgrade google-play-scraper'")
        return

    # Processing & Cleaning Data
    print("\n==================================================")
    print("[3] Memproses dan Menyimpan Data...")
    print("==================================================")
    data = []
    for item in results:
        rating = item.get('score', 0)
        
        # Abaikan rating 3 jika hanya ingin klasifikasi Positif & Negatif
        if rating == 3:
            continue
            
        label = 'positif' if rating >= 4 else 'negatif'
        
        data.append({
            'username': item.get('userName', ''),
            'rating': rating,
            'content': item.get('content', ''),
            'at': item.get('at').strftime('%Y-%m-%d %H:%M:%S') if item.get('at') else '',
            'sentiment': label
        })

    df = pd.DataFrame(data)
    
    # Cleaning
    df = df.dropna(subset=['content'])
    df = df[df['content'].astype(str).str.strip() != '']
    df = df.drop_duplicates(subset=['content'])

    output_file = 'edlink_scraped_reviews_full.csv'
    df.to_csv(output_file, index=False)
    
    print(f"[✓] BERHASIL! Disimpan {len(df)} data ulasan ke '{output_file}'")

if __name__ == '__main__':
    scrape_with_fallbacks()