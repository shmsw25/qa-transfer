class ResFileReader(object):
  """Process the result file from the search engine.

  This class can be extended to handle various file formats coming from 
  different IR engines, i.e. Terrier (answerbag) and Watson (jeopardy)
  The information we need to extract is qid, if the candidate answer is 
  relevant or not and the score IR engine assigned to this answer

  FIX: pretty up this class later on. 
  """
  def __init__(self, format="trec"):
    self.read_line = self.__getattribute__("read_line_%s" % format)

  def read_line_answerbag(self, line):
    tokens = line.strip().split()
    qid = tokens[0]
    aid = tokens[1]
    # this is the rank of the candidate, not the SE score, hence we invert it.
    ir_score = 1.0/int(tokens[2])
    relevant = tokens[3]  # true or false
    return qid, aid, relevant, ir_score

  def read_line_trec(self, line):
    """Process resultset where each line is in the TREC resultset format.

    Each line is formatted as follows:
      qid aid rank score relevance text
    """
    tokens = line.strip().split()
    qid = tokens[0]
    aid = tokens[1]
    # rank = int(tokens[2])
    ir_score = float(tokens[3])  # we invert the score
    relevant = tokens[4]  # true or false
    return qid, aid, relevant, ir_score
