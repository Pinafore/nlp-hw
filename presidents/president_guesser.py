import time

training_data = [
  {"start": 1789, "stop": 1797, "name": "George Washington"},
  {"start": 1797, "stop": 1801, "name": "John Adams"},
  {"start": 1801, "stop": 1809, "name": "Thomas Jefferson"},
  {"start": 1809, "stop": 1817, "name": "James Madison"},
  {"start": 1817, "stop": 1825, "name": "James Monroe"},
  {"start": 1825, "stop": 1829, "name": "John Quincy Adams"},
  {"start": 1829, "stop": 1837, "name": "Andrew Jackson"},
  {"start": 1837, "stop": 1841, "name": "Martin Van Buren"},
  {"start": 1841, "stop": 1841, "name": "William Henry Harrison"},
  {"start": 1841, "stop": 1845, "name": "John Tyler"},
  {"start": 1845, "stop": 1849, "name": "James K. Polk"},
  {"start": 1849, "stop": 1850, "name": "Zachary Taylor"},
  {"start": 1850, "stop": 1853, "name": "Millard Fillmore"},
  {"start": 1853, "stop": 1857, "name": "Franklin Pierce"},
  {"start": 1857, "stop": 1861, "name": "James Buchanan"},
  {"start": 1861, "stop": 1865, "name": "Abraham Lincoln"},
  {"start": 1865, "stop": 1869, "name": "Andrew Johnson"},
  {"start": 1869, "stop": 1877, "name": "Ulysses S. Grant"},
  {"start": 1877, "stop": 1881, "name": "Rutherford Birchard Hayes"},
  {"start": 1881, "stop": 1881, "name": "James A. Garfield"},
  {"start": 1881, "stop": 1885, "name": "Chester A. Arthur"},
  {"start": 1885, "stop": 1889, "name": "Grover Cleveland"},
  {"start": 1889, "stop": 1893, "name": "Benjamin Harrison"},
  {"start": 1893, "stop": 1897, "name": "Grover Cleveland"},
  {"start": 1897, "stop": 1901, "name": "William McKinley"},
  {"start": 1901, "stop": 1905, "name": "Theodore Roosevelt"},
  {"start": 1905, "stop": 1909, "name": "Theodore Roosevelt"},
  {"start": 1909, "stop": 1913, "name": "William H. Taft"},
  {"start": 1913, "stop": 1921, "name": "Woodrow Wilson"},
  {"start": 1921, "stop": 1923, "name": "Warren G. Harding"},
  {"start": 1923, "stop": 1929, "name": "Calvin Coolidge"},
  {"start": 1929, "stop": 1933, "name": "Herbert Hoover"},
  {"start": 1933, "stop": 1945, "name": "Franklin D. Roosevelt"},
  {"start": 1945, "stop": 1953, "name": "Harry S. Truman"},
  {"start": 1953, "stop": 1961, "name": "Dwight D. Eisenhower"},
  {"start": 1961, "stop": 1963, "name": "John F. Kennedy"},
  {"start": 1963, "stop": 1969, "name": "Lyndon B. Johnson"},
  {"start": 1969, "stop": 1974, "name": "Richard M. Nixon"},
  {"start": 1974, "stop": 1977, "name": "Gerald R. Ford"},
  {"start": 1977, "stop": 1981, "name": "Jimmy Carter"},
  {"start": 1981, "stop": 1989, "name": "Ronald Reagan"},
  {"start": 1989, "stop": 1993, "name": "George Bush"},
  {"start": 1993, "stop": 2001, "name": "Bill Clinton"},
  {"start": 2001, "stop": 2009, "name": "George W. Bush"},
  {"start": 2009, "stop": 2017, "name": "Barack Obama"},
  {"start": 2017, "stop": 2021, "name": "Donald J. Trump"},
  {"start": 2021, "stop": 2025, "name": "Joseph R. Biden"}
]

class PresidentGuesser:
    def train(self, training_data):
        self._lookup = {}
        
        for row in training_data:
            self._lookup[row["name"]] = (time.strptime(str(row["start"]), "%Y"), time.strptime(str(row["stop"]), "%Y"))
            
    def __call__(self, question):
        # Update this code so that we can have a different president than Joe
        # Biden
        candidates = ["Joseph R. Biden"]


        if len(candidates) == 0:
            return {"guess": ""}
        elif len(candidates) == 1:
            return {"guess": candidates[0]}
        
if __name__ == "__main__":
    pg = PresidentGuesser()

    pg.train(training_data)
    
    for date in ["Who was president on Wed Jan 25 06:20:00 2023?",
                     "Who was president on Sat May 23 02:00:00 1982?"]:
        print(date, pg(date)["guess"])
        
