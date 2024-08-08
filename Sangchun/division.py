import pandas as pd

# train.csv 파일 로드
train_file_path = './data/train.csv'
train_data = pd.read_csv(train_file_path)

# 공정 과정 관련 열 확인 및 분류
dam_dispenser_columns = [col for col in train_data.columns if 'Dam' in col]
fill1_columns = [col for col in train_data.columns if 'Fill1' in col]
fill2_columns = [col for col in train_data.columns if 'Fill2' in col]
auto_columns = [col for col in train_data.columns if 'Auto' in col]

# 각 공정 과정별 변수 개수
dam_dispenser_var_count = len(dam_dispenser_columns)
fill1_var_count = len(fill1_columns)
fill2_var_count = len(fill2_columns)
auto_var_count = len(auto_columns)

print(f"Dam Dispenser 변수 개수: {dam_dispenser_var_count}")
print(f"Fill1 변수 개수: {fill1_var_count}")
print(f"Fill2 변수 개수: {fill2_var_count}")
print(f"Auto 변수 개수: {auto_var_count}")

# 각 공정 과정별 결측값 및 존재하는 값 구분
def missing_values_info(columns):
    missing_info = train_data[columns].isnull().sum()
    existing_info = train_data[columns].notnull().sum()
    missing_columns = missing_info[missing_info > 0]
    existing_columns = existing_info[existing_info > 0]
    return missing_columns, existing_columns

dam_missing, dam_existing = missing_values_info(dam_dispenser_columns)
fill1_missing, fill1_existing = missing_values_info(fill1_columns)
fill2_missing, fill2_existing = missing_values_info(fill2_columns)
auto_missing, auto_existing = missing_values_info(auto_columns)

print("\nDam Dispenser - 결측값 있는 변수:")
print(dam_missing)
print("Dam Dispenser - 존재하는 값 있는 변수:")
print(dam_existing)

print("\nFill1 - 결측값 있는 변수:")
print(fill1_missing)
print("Fill1 - 존재하는 값 있는 변수:")
print(fill1_existing)

print("\nFill2 - 결측값 있는 변수:")
print(fill2_missing)
print("Fill2 - 존재하는 값 있는 변수:")
print(fill2_existing)

print("\nAuto - 결측값 있는 변수:")
print(auto_missing)
print("Auto - 존재하는 값 있는 변수:")
print(auto_existing)
