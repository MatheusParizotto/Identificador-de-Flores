import sqlite3

# Conectar ao banco de dados
conn = sqlite3.connect('data/cuidados.db')
cursor = conn.cursor()

# Informações sobre os cuidados das plantas
informacoes_de_cuidado = [
    ("Dente-de-leão", "Necessita de sol pleno e solo bem drenado. Regue moderadamente, deixando o solo secar entre as regas. Cresce bem em clima temperado. Pode ser invasiva, controle o crescimento."),
    ("Tulipa", "Precisa de solo bem drenado e deve ser plantada em locais com sol pleno ou parcial. Regue moderadamente, evitando encharcar. Plante os bulbos no outono e deixe as folhas murcharem após a floração."),
    ("Girassol", "Necessita de sol pleno e solo bem drenado. Regue regularmente, mantendo o solo úmido, mas sem encharcar. Plante as sementes com espaço suficiente e apoie as hastes em caso de vento forte."),
    ("Rosa", "Precisa de sol pleno e solo bem drenado e fértil. Regue regularmente, evitando molhar as folhas. Pode podar no inverno para estimular novas flores e controlar pragas."),
    ("Margarida", "Prefere sol pleno e solo bem drenado. Regue moderadamente, deixando o solo secar entre as regas. Remova flores murchas e faça a poda após a floração. Exige poucos cuidados.")
]

# Inserir os dados no banco de dados
for nome_planta, cuidados_necessarios in informacoes_de_cuidado:
    cursor.execute("INSERT INTO instrucoes (nome_planta, cuidados_necessarios) VALUES (?, ?)", (nome_planta, cuidados_necessarios))

conn.commit()
conn.close()

print("Informações de cuidado das plantas inseridas com sucesso!")
