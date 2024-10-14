import sqlite3

# Conexão com o banco
conn = sqlite3.connect('data/cuidados.db')

# Cria um cursor 
cursor = conn.cursor()

# Cria a tabela de cuidados de plantas
cursor.execute('''
CREATE TABLE IF NOT EXISTS instrucoes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome_planta TEXT NOT NULL,
    cuidados_necessarios TEXT NOT NULL
)
''')

# Salvar mudanças
conn.commit()

# Fechar conexão
conn.close()

print("Banco de dados criado com sucesso!")