;WITH ApptDetail AS 
(SELECT *, ROW_NUMBER() OVER (PARTITION BY TASK_ID ORDER BY CREATE_DTM DESC ) AS RNO
               FROM datahub.smart.limsAPT_DETAIL 
			   --WHERE TASK_ID ='14944213'
),
ApptScheduled AS 
(SELECT *, ROW_NUMBER() OVER (PARTITION BY TASK_ID ORDER BY CREATE_DTM ASC ) AS RNO
               FROM datahub.smart.limsAPT_DETAIL 
			   --WHERE TASK_ID ='15320118'
)
,ApptStatus AS
(
SELECT A.*, B.CS_ID,ROW_NUMBER() OVER (PARTITION BY B.CS_ID ORDER BY A.APT_DETAIL_ID ) AS RNO2,C.APT_STS_DESC, D.CREATE_DTM AS ScheduledDate
               FROM ApptDetail A
 LEFT JOIN datahub.smart.TASK B
 ON A.TASK_ID = B.TASK_ID
 LEFT JOIN datahub.smart.RT_APPOINTMENT_STATUS C
 ON A.APT_STS_ID = C.APT_STS_ID AND B.TASK_TYP_ID = C.TASK_TYP_ID
 LEFT JOIN ApptScheduled D
 ON A.TASK_ID = D.TASK_ID AND D.RNO = 1
 WHERE A.RNO = 1 
 --AND  A.TASK_ID ='14944213'
 --AND B.CS_ID = '1B8N651'
 )
 ,ApptCount AS
 (
 SELECT *
 ,LAG(A.RNO2) OVER (PARTITION BY A.CS_ID ORDER BY A.APT_DT) AS NumberofPastAppts 
 ,LAG(A.APT_DT) OVER (PARTITION BY A.CS_ID ORDER BY A.APT_DT) AS LastApptDate 
 FROM ApptStatus A 
 )
 ,NoShow AS 
 (
 SELECT *,ROW_NUMBER() OVER (PARTITION BY A.CS_ID ORDER BY A.APT_DETAIL_ID ) AS RNO3 FROM ApptStatus A 
 WHERE 
 --A.APT_STS_DESC = 'No Show' 
 --AND 
 A.CS_ID = '1B89J60'
 )
 ,ApptsHistory AS(
 SELECT A.APT_DETAIL_ID,A.CS_ID,A.TASK_ID,A.NumberofPastAppts,A.LastApptDate,MAX(B.RNO3) AS PastNoShow,A.ScheduledDate FROM ApptCount A 
 LEFT JOIN NoShow B
 ON A.CS_ID = B.CS_ID AND A.APT_DT > B.APT_DT
 -- WHERE A.CS_ID = '1B65D39'
 GROUP BY A.APT_DETAIL_ID,A.CS_ID,A.TASK_ID,A.NumberofPastAppts,A.LastApptDate,A.ScheduledDate)
 
 --SELECT * FROM datahub.LIMs.vwCurrentAppointmentDetails WHERE  AppointmentDate >= DATEADD(MONTH, -8, GETDATE())

   SELECT 
   A.*
   ,RT.DayNumberOfWeek
   ,RT.DayNumberOfMonth
   ,RT.DayNumberOfYear
   ,RT.IsHoliday
   ,RT.MonthNumberOfYear
   ,DATEPART(HOUR,A.AppointmentStartTime) AS AppHour
   ,DATEDIFF(YEAR,C.DOB,GETDATE()) AS Age,C.Sex,C.HomeZip,C.HomeZip4
	,D.NumberofPastAppts
	,D.LastApptDate
	,D.ScheduledDate
	,D.PastNoShow
	,DATEDIFF(DAY,D.LastApptDate,A.AppointmentDate) Interval
	,DATEDIFF(DAY,D.ScheduledDate,A.AppointmentDate) Waittime
 FROM datahub.LIMs.vwCurrentAppointmentDetails A
	LEFT JOIN DHABusinessIntelligence.Configuration.DimDate RT
	ON A.AppointmentDate = RT.FullDateAlternateKey
	LEFT JOIN datahub.WelfareProgram.vwCase B
	ON A.CaseID = B.CaseID 
	LEFT JOIN datahub.WelfareIndividual.vwClient C
	ON B.HeadofHouseholdCWIN = C.CWIN
	LEFT JOIN ApptsHistory D
	ON A.TaskID = D.TASK_ID
	--WHERE DATEDIFF(DAY,D.LastApptDate,A.AppointmentDate) >= 30000
	WHERE A.CaseID = '1B89J60'
	WHERE DATEPART(HOUR,A.AppointmentStartTime) = 23
	AND DATEDIFF(DAY,D.LastApptDate,A.AppointmentDate) < 0
 --WHERE A.AppointmentDate >= DATEADD(MONTH, -8, GETDATE())
 --AND A.CaseID = '1B89J60'

 SELECT * FROM smart.limsTIME_SLOT
 SELECT * FROM smart.SUB_TASK_TYPE
 WHERE SUB_TASK_TYP_ID IN (209,210,212)

 SELECT * FROM smart.TASK_TYPE WHERE TASK_TYP_ID IN
(
162,
163
)


 --SELECT * FROM smart.[GROUP]

 --SELECT TOP 100 * FROM datahub.WelfareIndividual.vwClient
  
 --SELECT TOP 100* FROM datahub.EmploymentServices.ClientAddress WHERE Address1 LIKE '%5941 BELLEVIEW%'


SELECT A.*, B.CS_ID FROM datahub.smart.limsAPT_DETAIL A  
LEFT JOIN datahub.smart.TASK B
ON A.TASK_ID = B.TASK_ID
WHERE B.CS_ID = '1B89J60'
ORDER BY A.CREATE_DTM

SELECT * FROM datahub.smart.RT_APPOINTMENT_STATUS WHERE APT_STS_ID IN (17,18)

SELECT * FROM DHAManagementReporting.MRAutomationReport.DepartmentsReportDetail
WHERE ProgramCD = 'WW'

SELECT DISTINCT A.CaseID FROM [DHAManagementReporting].[MRAutomation].[WR_ES_WTWOverViewAllStatus] A 
WHERE ReportMonth = '2018-12-01'
and A.caseID NOT IN 
(SELECT CaseID FROM DHAManagementReporting.MRAutomationReport.DepartmentsReportDetail
WHERE ProgramCD = 'WW')
AND A.CaseID ='1B88S41'



SELECT CaseID FROM DHAManagementReporting.MRAutomationReport.DepartmentsReportDetail
WHERE ProgramCD = 'WW'
AND CaseID ='1B88S41'

--1B3C574
--1B88S41

SELECT * FROM WelfareProgram.vwCaseStatus 
WHERE CaseID IN
(
'1B1W404',
'1B1SZ26',
'1B34Q18',
'B714027'
)

AND ProgramCD ='WW'
ORDER BY CaseID,ProgramStatusDate



;WITH ApptDetail AS 
(SELECT *, ROW_NUMBER() OVER (PARTITION BY TASK_ID ORDER BY CREATE_DTM ASC ) AS RNO
               FROM datahub.smart.limsAPT_DETAIL 
			   --WHERE TASK_ID ='15320118'
)
SELECT  A.*
, B.CS_ID,B.TASK_TYP_ID,D.TASK_TYP_DESC,B.SUB_TASK_TYP_ID,E.SUB_TASK_TYP_DESC,F.TIME_START,F.TIME_END
               FROM ApptDetail A
 LEFT JOIN datahub.smart.TASK B
 ON A.TASK_ID = B.TASK_ID
 LEFT JOIN datahub.smart.RT_APPOINTMENT_STATUS C
 ON A.APT_STS_ID = C.APT_STS_ID AND B.TASK_TYP_ID = C.TASK_TYP_ID
 LEFT JOIN smart.TASK_TYPE  D
 ON B.TASK_TYP_ID = D.TASK_TYP_ID
 LEFT JOIN smart.SUB_TASK_TYPE E
 ON  B.SUB_TASK_TYP_ID = E.SUB_TASK_TYP_ID
 LEFT JOIN smart.limsTIME_SLOT F
 ON A.TIME_SLOT_ID = F.TIME_SLOT_ID
 WHERE A.RNO = 1 
 --AND A.TIME_SLOT_ID IS NULL
 AND B.TASK_TYP_ID IN (162,163)




 SELECT * FROM [LIMs].[vwCurrentAppointmentDetails]

  SELECT * FROM smart.limsTIME_SLOT WHERE LOBBY_BUREAU_ID = 7

  SELECT DISTINCT LOBBY_BUREAU_ID FROM  datahub.smart.limsAPT_DETAIL 
  SELECT * FROM [SMART].[cnfgLOBBY_BUREAU]
  SELECT * FROM [DataHub].[SMART].[limsRT_LOBBY]

  IN
(
'CR',
'FS',
'GA',
'IN',
'EX',
'MC'
)


    --SELECT * FROM [SMART].[TASK_PROGRAMS] WHERE CREATE_DTM >= '2018-01-01' AND IS_SELECTED = 1

  WITH programtable AS (
  SELECT TASK_ID, CreateDate,
[CA],
[CC],
[CL],
[CM],
[CN],
[CR],
[DV],
[ER],
[ES],
[EX],
[FC],
[FS],
[FT],
[GA],
[GE],
[GF],
[HA],
[IN],
[KG],
[M1],
[M2],
[M3],
[MC],
[PH],
[SC],
[TH],
[WW]
FROM
  (SELECT A.TASK_ID, cast ( A.CREATE_DTM AS DATE) AS CreateDate,A.PGM_CD,CAST(A.IS_SELECTED AS INT) AS SELECTED FROM [SMART].[TASK_PROGRAMS] A ) P
  PIVOT
  (SUM(SELECTED) FOR PGM_CD IN
(
[CA],
[CC],
[CL],
[CM],
[CN],
[CR],
[DV],
[ER],
[ES],
[EX],
[FC],
[FS],
[FT],
[GA],
[GE],
[GF],
[HA],
[IN],
[KG],
[M1],
[M2],
[M3],
[MC],
[PH],
[SC],
[TH],
[WW]
)

 ) AS pvt)

 ,ApptDetail AS 
(SELECT *, ROW_NUMBER() OVER (PARTITION BY TASK_ID ORDER BY CREATE_DTM ASC ) AS RNO
               FROM datahub.smart.limsAPT_DETAIL 
			   --WHERE TASK_ID ='15320118'
)
SELECT  A.*
, B.CS_ID,B.TASK_TYP_ID,D.TASK_TYP_DESC,B.SUB_TASK_TYP_ID,E.SUB_TASK_TYP_DESC,F.TIME_START,F.TIME_END, G.*
               FROM ApptDetail A
 LEFT JOIN datahub.smart.TASK B
 ON A.TASK_ID = B.TASK_ID
 LEFT JOIN datahub.smart.RT_APPOINTMENT_STATUS C
 ON A.APT_STS_ID = C.APT_STS_ID AND B.TASK_TYP_ID = C.TASK_TYP_ID
 LEFT JOIN smart.TASK_TYPE  D
 ON B.TASK_TYP_ID = D.TASK_TYP_ID
 LEFT JOIN smart.SUB_TASK_TYPE E
 ON  B.SUB_TASK_TYP_ID = E.SUB_TASK_TYP_ID
 LEFT JOIN smart.limsTIME_SLOT F
 ON A.TIME_SLOT_ID = F.TIME_SLOT_ID
 LEFT JOIN programtable G
 ON a.TASK_ID = g.TASK_ID
 WHERE A.RNO = 1 
 AND G.AA IS NOT NULL

  8624160