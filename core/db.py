import mysql.connector
import mysqlx
from core.feature import *
from file_cache.utils.util_log import timed




import contextlib


version = 3

def get_connect():
    db = mysql.connector.connect(user='ai_lab', password=mysql_pass,
                                 host='vm-ai-2',
                                 database='ai')
    return db


def insert(row):
    if len(row) == 0:
        return

    db = get_connect()
    # logger.info(f'merge_2days:{merge_2days.columns}')
    try:
        sql = """insert into metro_score(
                    day ,
                    week_day   ,
                    direct,
                    stationID  ,
                    time_ex    ,
                    obs      ,
                    predict   ,
                    p,
                    d,
                    q
                    )
                    values
                    (    
                    {day},       
                    {week_day}, 
                    '{direct}',
                    {stationID},
                    {time_ex},
                    {obs},
                    {predict},
                    {p},
                    {d},
                    {q}            
                   )
                    """.format(**row, version=version)
        cur = db.cursor()
        logger.debug(sql)
        cur.execute(sql)
        db.commit()
    except mysql.connector.IntegrityError as e:
        logger.warning(f'IntegrityError: for {row}')




if __name__ == '__main__' :
    insert(['2019-01-12', '2019-01-19'])
    insert(['2019-01-13', '2019-01-20'])
    insert(['2019-01-15', '2019-01-22'])
