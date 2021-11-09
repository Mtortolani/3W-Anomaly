from Data_Processing import DataProcessor
from Model_Creation import ModelCreator

Processor = DataProcessor()
Processor.total_data_compiler()
print(Processor.total_data[0].shape)


