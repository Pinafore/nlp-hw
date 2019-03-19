from typing import List, Dict, Iterable, Optional, Tuple, NamedTuple
import os
import json


class Question(NamedTuple):
    qanta_id: int
    text: str
    first_sentence: str
    tokenizations: List[Tuple[int, int]]
    answer: str
    page: Optional[str]
    fold: str
    gameplay: bool
    category: Optional[str]
    subcategory: Optional[str]
    tournament: str
    difficulty: str
    year: int
    proto_id: Optional[int]
    qdb_id: Optional[int]
    dataset: str

    def to_json(self) -> str:
        return json.dumps(self._asdict())

    @classmethod
    def from_json(cls, json_text):
        return cls(**json.loads(json_text))

    @classmethod
    def from_dict(cls, dict_question):
        return cls(**dict_question)

    def to_dict(self) -> Dict:
        return self._asdict()

    @property
    def sentences(self) -> List[str]:
        """
        Returns a list of sentences in the question using preprocessed spacy 2.0.11
        """
        return [self.text[start:end] for start, end in self.tokenizations]

    def runs(self, char_skip: int) -> Tuple[List[str], List[int]]:
        """
        A Very Useful Function, especially for buzzer training.
        Returns runs of the question based on skipping char_skip characters at a time. Also returns the indices used
        q: name this first united states president.
        runs with char_skip=10:
        ['name this ',
         'name this first unit',
         'name this first united state p',
         'name this first united state president.']
        :param char_skip: Number of characters to skip each time
        """
        char_indices = list(range(char_skip, len(self.text) + char_skip, char_skip))
        return [self.text[:i] for i in char_indices], char_indices


class QantaDatabase:
    def __init__(self, split):   
        '''
        split can be {'train', 'dev', 'test'} - gets both the buzzer and guesser folds from the corresponding data file.
        '''
        dataset_path = os.path.join('..', 'qanta.'+split+'.json')
        with open(dataset_path) as f:
            self.dataset = json.load(f)

        self.version = self.dataset['version']
        self.raw_questions = self.dataset['questions']
        self.all_questions = [Question(**q) for q in self.raw_questions]
        self.mapped_questions = [q for q in self.all_questions if q.page is not None]
        
        self.guess_questions = [q for q in self.mapped_questions if q.fold == 'guess'+split]
        self.buzz_questions = [q for q in self.mapped_questions if q.fold == 'buzz'+split]




class QuizBowlDataset:
    def __init__(self, *, guesser = False, buzzer = False, split='train'):
        """
        Initialize a new quiz bowl data set
        guesser = True/False -> to use data from the guesser fold or not
        buzzer = True/False -> to use data from the buzzer fold or not
        split can be {'train', 'dev', 'test'} 
        Together, these three parameters (two bools and one str) specify which specific fold's data to return - 'guesstrain'/'buzztrain'/'guessdev'/'buzzdev'/'guesstest'/'buzztest'
        """
        super().__init__()
        if not guesser and not buzzer:
            raise ValueError('Requesting a dataset which produces neither guesser or buzzer training data is invalid')

        if guesser and buzzer:
            print('Using QuizBowlDataset with guesser and buzzer training data, make sure you know what you are doing!')

        self.db = QantaDatabase(split)
        self.guesser = guesser
        self.buzzer = buzzer

    def data(self):
        '''
        Returns the questions - where each question is an object of the Question class - according to the specific fold specified by the split, guesser, buzzer parameters.
        '''
        questions = []
        if self.guesser:
            questions.extend(self.db.guess_questions)
        if self.buzzer:
            questions.extend(self.db.buzz_questions)
        
        return questions

