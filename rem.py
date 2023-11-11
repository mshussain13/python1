import os
import filecmp
import shutil
'''
# for camparing two folder of images and find duplicate

def find_duplicate_files(folder1, folder2, output_folder):
    duplicates = []

    # Iterate over files in folder1
    for root, _, files in os.walk(folder1):
        for file in files:
            file1 = os.path.join(root, file)
            file2 = os.path.join(folder2, os.path.relpath(file1, folder1))

            # Check if file exists in folder2 and if they are duplicates
            if os.path.isfile(file2) and filecmp.cmp(file1, file2):
                duplicates.append(file1)

    # Move duplicate files to output folder
    for file in duplicates:
        destination = os.path.join(output_folder, os.path.basename(file))
        shutil.move(file, destination)
        print(f"Moved {file} to {destination}")

# Specify the folders to compare
folder1 = '/home/shussain/Downloads/test1'
folder2 = '/home/shussain/Downloads/test2'

# Specify the output folder for duplicate files
output_folder = '/home/shussain/Downloads/test3'

# Call the function
find_duplicate_files(folder1, folder2, output_folder)

'''
'''
import os
import filecmp
import shutil

def find_duplicate_files(folder1, folder2, output_folder):
    duplicates = []

    # Iterate over files in folder1
    for root, _, files in os.walk(folder1):
        for file in files:
            if file.lower().endswith('.txt') and file.lower().endswith('.jpg'):
                file1 = os.path.join(root, file)
                file2 = os.path.join(folder2, os.path.relpath(file1, folder1))

                # Check if file exists in folder2 and if they are duplicates
                if os.path.isfile(file2) and filecmp.cmp(file1, file2):
                    duplicates.append(file1)

    # Move duplicate files to output folder
    for file in duplicates:
        destination = os.path.join(output_folder, os.path.basename(file))
        shutil.move(file, destination)
        print(f"Moved {file} to {destination}")

# Specify the folders to compare
folder1 = '/home/shussain/Downloads/test1'
folder2 = '/home/shussain/Downloads/test2'

# Specify the output folder for duplicate files
output_folder = '/home/shussain/Downloads/test3'

# Call the function
find_duplicate_files(folder1, folder2, output_folder)

'''
'''
class office():
    def __init__(self,emp):
        self.emp = emp
    def work(self):
        print(self.emp + 'is working')

my_office = office('ms')

print(my_office.emp +" "+ "is a great employee")
my_office.work

fav_languages = {
'jen': ['python', 'ruby'],
'sarah': ['c'],
'edward': ['ruby', 'go'],
'phil': ['python', 'haskell'],
}
# Show all responses for each person.
for name, langs in fav_languages.items():
    print(name + ": ")
    for lang in langs:
        print("- " + lang)
        
'''
aliens = []
# Make a million green aliens, worth 5 points
# each. Have them all start in one row.
for alien_num in range(1000000):
    new_alien = {}
    new_alien['color'] = 'green'
    new_alien['points'] = 5
    new_alien['x'] = 20 * alien_num
    new_alien['y'] = 0
    aliens.append(new_alien)
# Prove the list contains a million aliens.
num_aliens = len(aliens)
print("Number of aliens created:")
print(num_aliens)


