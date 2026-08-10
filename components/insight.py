"""
Insight.py Nazalyst
"""

import streamlit as st


def sentiment_insight(result):

    positif=result["positif_percent"]

    negatif=result["negatif_percent"]

    netral=result["netral_percent"]

    if negatif>positif:

        st.error(

f"""
Mayoritas ulasan bersifat **Negatif**
({negatif:.2f}%).

Hal ini menunjukkan bahwa pengguna
masih banyak mengalami kendala ketika
menggunakan aplikasi MyPertamina.
"""

        )

    else:

        st.success(

f"""
Mayoritas ulasan bersifat **Positif**
({positif:.2f}%).

Hal ini menunjukkan bahwa pengguna
cukup puas terhadap aplikasi.
"""

        )

    if netral<1:

        st.info(

"""
Jumlah sentimen netral sangat sedikit.

Sebagian besar pengguna secara jelas
memberikan opini positif atau negatif.
"""

        )