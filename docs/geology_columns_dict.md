
text below copied from https://earthwise.bgs.ac.uk/index.php/OR/16/046_Technical_information - the tech report for the 50k resolution version of the 625k geology datafile

---



Table 1    Attribution of bedrock and superficial themes (GIS Layers)

Data Field	Explanation of Data Field	Note
LEX_WEB	Direct hyperlink to the definition of the particular geological unit in the BGS Lexicon of Named Rock Units
(BGS website): e.g. www.bgs.ac.uk/Lexicon/lexicon.cfm?pub=GOG	Note 1
LEX	A single Lexicon (or LEX) computer code used to identify the rock unit(s) or deposit(s) as listed in the BGS Lexicon of Named Rock Units: e.g. GOG	Note 2
LEX_D	A description of the LEX code above, giving the full name of the unit(s): e.g. GREAT OOLITE GROUP is the full name of the unit coded as GOG
LEX_RCS	The primary two-part, LEX & RCS, code used to label the geological units in BGS Geology data: e.g. GOG-LMST	Note 3
RCS	A rock-classification code of up to 6 characters (mostly letters) forming the second part of the primary LEX_RCS attribute. e.g. MDCO. The code can represent a single lithology or multiple lithology’s (see RCS_X)	Note 4
RCS_X	A variant of the RCS code (above) which individually lists the components of heterolithic units: e.g. MDST + [CONG] (shown as RCS = MDCO). Subordinate units are denoted in [ ] brackets	Note 5
RCS_D	Description of the RCS code(s) above giving the lithology of the unit: e.g. MUDSTONE AND [SUBEQUAL/SUBORDINATE] CONGLOMERATE is the description of the rock coded as MDST + [CONG]
RCS_ORIGIN	An attribute of the RCS code(s) above, classifying the mode of origin of the lithology of the rock/deposit: e.g. Sedimentary, Igneous, and Metamorphic	Note 6
RANK	Rank of the unit in the lithostratigraphical or lithodemic hierarchy: e.g. BED or SUITE	Note 7
BED_EQ_D	Description of the Bed or equivalent lexicon code for the unit where applicable
MB_EQ_D	Description of the Member or equivalent lexicon code for the unit where applicable
FM_EQ_D	Description of the Formation or equivalent lexicon code for the unit where applicable
SUBGP_EQ_D	Description of the Sub-Group or equivalent lexicon code for the unit where applicable
GP_EQ_D	Description of the Group or equivalent lexicon code for the unit where applicable
SUPGP_EQ_D	Description of the Super-Group or equivalent lexicon code for the unit where applicable
MAX_TIME_Y	Maximum age (in years), of the oldest time division in which the geological unit was formed: e.g. 170300000	Note 8
MIN_TIME_Y	Minimum age (in years), of the youngest time division in which the geological unit was formed: e.g. 163500000
MAX_AGE	Maximum age defined for the unit e.g. ASBIAN	Note 9
MAX_EPOCH	Maximum epoch defined for the unit: e.g. VISEAN
MAX_SUBPER	Maximum sub-period defined for the unit: e.g. MISSISSIPPIAN
MAX_ PERIOD	Maximum period defined for the unit e.g. CARBONIFEROUS
MAX_ERA	Maximum era defined for the unit e.g. PALAEOZOIC
MAX_EON	Maximum eon defined for the unit e.g. PROTEROZOIC
BGSTYPE	The BGS Geology theme: e.g. BEDROCK, SUPERFICIAL
LEX_RCS_I	A computer code that can be used to sort units into approximately the correct stratigraphical order (by Period).
NB it does not completely resolve UK stratigraphy and must NOT be used as a substitute for determining full stratigraphical relationships between units.
LEX_RCS_D	A full description of the LEX_RCS above: e.g. GREAT OOLITE GROUP — LIMESTONE
BGSREF	A BGS code used to define the colour for the polygon based on the LEX_RCS code pair. Colour information can now be applied from ‘add on’ tables in a variety of ways, please see Appendix 4
MAP_SRC	Name of the digital geological tile (number and name based on published map sheet) that the polygon appears on: e.g. EW075_PRESTON, SC084E_NAIRN where prefix ‘EW’ is for England & Wales and ‘SC’ for Scotland	Note 10
MAP_WEB	The MAP_WEB link provides a direct hyperlink to the appropriate, original, paper maps held in the BGS Map Portal www.bgs.ac.uk/data/maps/home.html	Note 11
OS_TILE	Ordnance Survey 5 km tile identifier. This is used to enable BGS Geology products to be updated in 5km tiles and allow integration into best-available scale maps (only available in the variant Ordnance Survey (OS) 5 km tiled version of the dataset)	New
VERSION	Version number and attribute level of the digital data: e.g. 8.24 is version 8, with attribute level 24	
RELEASED	Date the BGS Geology data files were created by BGS: e.g. 28-07-2016	
NOM_SCALE	Nominal scale of the published (or compiled) information used to prepare the digital data: e.g. 50000 for 1:50 000 [including 1:63 360 and 1:100 000 maps] (see limitations section below)	
NOM_BGS_YR	The year date of publication of the most up-to-date map sheet, or the date of publication in BGS Geology: 50k
(if no map previously exists). Where not known or inappropriate, field is null	
UUID	Universally Unique Identification that can be used to identify individual features: e.g. bgsn:DM50_V8_digmap1004081046355357	

Fields in GREEN are derived from the BGS Lexicon	Fields in BLUE are derived from the BGS Rock Classification Scheme
Fields in PURPLE are derived from the BGS Geological timechart	Fields in brown are for metadata purposes

Note 1	The LEX_WEB link provides a hyperlink to the online Lexicon resource. The online version is updated regularly.
Note 2	The Lex attribute is the computer code linking to the BGS Lexicon (database of named rock units) www.bgs.ac.uk/lexicon/home.cfm.The Lexicon code may refer to a single identifiable unit or a package of units where the individual components cannot be differentiated.
Note 3	The BGS Geology dataset uses the LEX_RCS codes as a primary key, which can be used to JOIN (append) ‘add-on’ datasets
Note 4	The RCS attribute is the computer code linking to the BGS Rock Classification Scheme (RCS) www.bgs.ac.uk/bgsrcs/home.html The field may include abbreviated codes for multiple lithologies
Note 5	The RCS_X field provides a list of individual RCS lithology codes that make up the overall lithological description of the unit. The suffix _X was added to distinguish this listing of the components from the abbreviated code now shown in the RCS field.
Note 6	The origin of each rock/deposit type has been introduced in version 8, this is in part to assist users who wanted to know some fundamental properties of the geological materials (see future revisions section below).
Note 7	The parentage of each rock/deposit is provided in these fields and these are all derived from the BGS LEXICON. The ‘RANK’ of a unit identifies the units position within a hierarchy eg a ‘bed’ may be part of a named member, which is itself part of a formation, several formations may make up a group and several groups may form a supergroup. The BED, MEMBER, FORMATION, SUBGRP GROUP and SUPGRP codes/names describe the ascending parentage for each unit (other non-stratigraphic schema are also shown in this way). A formation is the fundamental lithostratoigraphical unit and is the prime mapping-unit for BGS and need not be divided up into named members or beds; nor does a formation have to belong to a group or supergroup. ‘NotAp’ is the abbreviation for ‘Not Applicable’ and is used to indicate that it is not appropriate to list child units of lower rank; ‘NoPar’ is the abbreviation for ‘No Parent’ and is used to indicate that no parental unit of higher rank has been identified. Users are recommended to refer to the LEX_WEB link to find the latest information concerning the lithostratigraphy of a unit. All codes and names used in V8 are correct at time of publication.
Note 8	These figures give an indication of the maximum age range (in years before present) of the lithostratigraphical units as given in the BGS Geological Timechart available at: www.bgs.ac.uk/downloads/browse.cfm?sec=8&cat=39 (where they are expressed as ‘million years’). Some of these values are interpolations; the +/- error ranges are not provided here. The age range given is that for the time period ascribed to each geological unit in the BGS Lexicon. They do not give absolute age measurements made on the individual geological units (see future revisions section below).
Note 9	The maximum geochronological age (expressed as age/stage/chron, epoch, sub-period, period, era or eon) for each rock/deposit is provided in these fields. These are all derived from the BGS Lexicon and Geological Timechart. ‘NOT DEFINED’ is used to indicate that no age classification has been identified (or is needed). Users are recommended to refer to the LEX_WEB link to find the latest information concerning the lithostratigraphy of a unit. Some geological units straddle more than one geological age.
All codes and names used in V8 are correct at time of publication.

Note 10	This attribute was previously called SHEET. It has been changed in Version 8 to MAP_SRC, to reflect that BGS Geology is no longer just compiled from published map sheets, but from a range of sources.
Note 11	The MAP_WEB link provides a hyperlink to any online resource that acts as reference material for BGS Geology content. Currently, the weblink will take users to the appropriate, original, paper maps held in the BGS Map Portal www.bgs.ac.uk/data/maps/home.html (future versions will hyperlink to other resources).