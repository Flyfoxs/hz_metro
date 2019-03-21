

drop table metro_score;

create table metro_score(
ID int NOT NULL AUTO_INCREMENT PRIMARY KEY,
direct VARCHAR(5),
day int,
stationID int,
time_ex int,
p int,
d int,
q int,
bin_id	int,

week_day int,

obs DECIMAL(8,2),
predict DECIMAL(8,2),
version int,

ct     TIMESTAMP DEFAULT NOW(),
mt TIMESTAMP DEFAULT  now()
);

CREATE UNIQUE INDEX pk_metro_score on metro_score(
direct,
day  ,
stationID  ,
time_ex  ,
p  ,
d  ,
q  );

CREATE INDEX day_index
ON metro_score (day);


CREATE INDEX week_day_index
ON metro_score (week_day);

CREATE INDEX stationID_index
ON metro_score (stationID);

CREATE INDEX time_ex_index
ON metro_score (time_ex);

CREATE INDEX bin_id_index
ON metro_score (bin_id);
