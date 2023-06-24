import enum

@enum.unique
class Task(enum.Enum):
  """Look up for tasks."""
  
  NEXT_ACTIVITY = "next_activity"
  NEXT_TIME = "next_time"
  REMAINING_TIME = "remaining_time"
  OUTCOME_ORIENTED = "outcome_oriented"

@enum.unique
class Dataset(enum.Enum):
  HELPDESK = "helpdesk"
  BPIC2017 = "BPIC2017"
  BPIC2011 = "BPIC2011"
  BPIC2012 = "BPIC2012"
  BPIC2015 = "BPIC2015"