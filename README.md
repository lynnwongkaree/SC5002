# SC5002 Lab Assignment 2
Exploring Linear and Ridge Regression with Cross-Validation 
By: Frank, Cadence, Lynn

## Objectives
The goal of this project is to explore the differences between Linear Regression and Ridge Regression by using the dataset [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). We aim to compare the strengths and weaknesses of both models to improve their performance.


## Dataset
The dataset we used is Kaggle's [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). 

The Ames Housing dataset contains detailed information on residential homes in Ames, Iowa. It contains 79 explanatory variables describing almost every aspect of a house and 1460 examples, each representing an individual residential property. This dataset provides a realistic, complex regression problem that combines both numerical and categorical predictors. 


<details>
<summary> Dataset Dictionary </summary>
  
<table>
  <tr>
    <th>Feature</th>
    <th colspan="2">Description / Categories</th>
  </tr>

  <!-- MSSubClass -->
  <tr>
    <td><b>MSSubClass</b></td>
    <td colspan="2">Identifies the type of dwelling involved in the sale.</td>
  </tr>
  <tr><td></td><td>20</td><td>1-STORY 1946 &amp; NEWER ALL STYLES</td></tr>
  <tr><td></td><td>30</td><td>1-STORY 1945 &amp; OLDER</td></tr>
  <tr><td></td><td>40</td><td>1-STORY W/FINISHED ATTIC ALL AGES</td></tr>
  <tr><td></td><td>45</td><td>1-1/2 STORY - UNFINISHED ALL AGES</td></tr>
  <tr><td></td><td>50</td><td>1-1/2 STORY FINISHED ALL AGES</td></tr>
  <tr><td></td><td>60</td><td>2-STORY 1946 &amp; NEWER</td></tr>
  <tr><td></td><td>70</td><td>2-STORY 1945 &amp; OLDER</td></tr>
  <tr><td></td><td>75</td><td>2-1/2 STORY ALL AGES</td></tr>
  <tr><td></td><td>80</td><td>SPLIT OR MULTI-LEVEL</td></tr>
  <tr><td></td><td>85</td><td>SPLIT FOYER</td></tr>
  <tr><td></td><td>90</td><td>DUPLEX - ALL STYLES AND AGES</td></tr>
  <tr><td></td><td>120</td><td>1-STORY PUD (Planned Unit Development) - 1946 &amp; NEWER</td></tr>
  <tr><td></td><td>150</td><td>1-1/2 STORY PUD - ALL AGES</td></tr>
  <tr><td></td><td>160</td><td>2-STORY PUD - 1946 &amp; NEWER</td></tr>
  <tr><td></td><td>180</td><td>PUD - MULTILEVEL - INCL SPLIT LEV/FOYER</td></tr>
  <tr><td></td><td>190</td><td>2 FAMILY CONVERSION - ALL STYLES AND AGES</td></tr>

  <!-- MSZoning -->
  <tr>
    <td><b>MSZoning</b></td>
    <td colspan="2">Identifies the general zoning classification of the sale.</td>
  </tr>
  <tr><td></td><td>A</td><td>Agriculture</td></tr>
  <tr><td></td><td>C</td><td>Commercial</td></tr>
  <tr><td></td><td>FV</td><td>Floating Village Residential</td></tr>
  <tr><td></td><td>I</td><td>Industrial</td></tr>
  <tr><td></td><td>RH</td><td>Residential High Density</td></tr>
  <tr><td></td><td>RL</td><td>Residential Low Density</td></tr>
  <tr><td></td><td>RP</td><td>Residential Low Density Park</td></tr>
  <tr><td></td><td>RM</td><td>Residential Medium Density</td></tr>

  <!-- LotFrontage -->
  <tr>
    <td><b>LotFrontage</b></td>
    <td colspan="2">Linear feet of street connected to property.</td>
  </tr>

  <!-- LotArea -->
  <tr>
    <td><b>LotArea</b></td>
    <td colspan="2">Lot size in square feet.</td>
  </tr>

  <!-- Street -->
  <tr>
    <td><b>Street</b></td>
    <td colspan="2">Type of road access to property.</td>
  </tr>
  <tr><td></td><td>Grvl</td><td>Gravel</td></tr>
  <tr><td></td><td>Pave</td><td>Paved</td></tr>

  <!-- Alley -->
  <tr>
    <td><b>Alley</b></td>
    <td colspan="2">Type of alley access to property.</td>
  </tr>
  <tr><td></td><td>Grvl</td><td>Gravel</td></tr>
  <tr><td></td><td>Pave</td><td>Paved</td></tr>
  <tr><td></td><td>NA</td><td>No alley access</td></tr>

  <!-- LotShape -->
  <tr>
    <td><b>LotShape</b></td>
    <td colspan="2">General shape of the property.</td>
  </tr>
  <tr><td></td><td>Reg</td><td>Regular</td></tr>
  <tr><td></td><td>IR1</td><td>Slightly Irregular</td></tr>
  <tr><td></td><td>IR2</td><td>Moderately Irregular</td></tr>
  <tr><td></td><td>IR3</td><td>Irregular</td></tr>

  <!-- LandContour -->
  <tr>
    <td><b>LandContour</b></td>
    <td colspan="2">Flatness of the property.</td>
  </tr>
  <tr><td></td><td>Lvl</td><td>Near Flat/Level</td></tr>
  <tr><td></td><td>Bnk</td><td>Banked – Quick and significant rise from street grade to building</td></tr>
  <tr><td></td><td>HLS</td><td>Hillside – Significant slope from side to side</td></tr>
  <tr><td></td><td>Low</td><td>Depression</td></tr>

  <!-- Utilities -->
  <tr>
    <td><b>Utilities</b></td>
    <td colspan="2">Type of utilities available.</td>
  </tr>
  <tr><td></td><td>AllPub</td><td>All public utilities (Electricity, Gas, Water, Sewer)</td></tr>
  <tr><td></td><td>NoSewr</td><td>Electricity, Gas, and Water (Septic Tank)</td></tr>
  <tr><td></td><td>NoSeWa</td><td>Electricity and Gas Only</td></tr>
  <tr><td></td><td>ELO</td><td>Electricity Only</td></tr>

  <!-- LotConfig -->
  <tr>
    <td><b>LotConfig</b></td>
    <td colspan="2">Lot configuration.</td>
  </tr>
  <tr><td></td><td>Inside</td><td>Inside lot</td></tr>
  <tr><td></td><td>Corner</td><td>Corner lot</td></tr>
  <tr><td></td><td>CulDSac</td><td>Cul-de-sac</td></tr>
  <tr><td></td><td>FR2</td><td>Frontage on 2 sides of property</td></tr>
  <tr><td></td><td>FR3</td><td>Frontage on 3 sides of property</td></tr>

  <!-- LandSlope -->
  <tr>
    <td><b>LandSlope</b></td>
    <td colspan="2">Slope of the property.</td>
  </tr>
  <tr><td></td><td>Gtl</td><td>Gentle slope</td></tr>
  <tr><td></td><td>Mod</td><td>Moderate slope</td></tr>
  <tr><td></td><td>Sev</td><td>Severe slope</td></tr>

  <!-- Neighborhood -->
  <tr>
    <td><b>Neighborhood</b></td>
    <td colspan="2">Physical locations within Ames city limits.</td>
  </tr>
  <tr><td></td><td>Blmngtn</td><td>Bloomington Heights</td></tr>
  <tr><td></td><td>Blueste</td><td>Bluestem</td></tr>
  <tr><td></td><td>BrDale</td><td>Briardale</td></tr>
  <tr><td></td><td>BrkSide</td><td>Brookside</td></tr>
  <tr><td></td><td>ClearCr</td><td>Clear Creek</td></tr>
  <tr><td></td><td>CollgCr</td><td>College Creek</td></tr>
  <tr><td></td><td>Crawfor</td><td>Crawford</td></tr>
  <tr><td></td><td>Edwards</td><td>Edwards</td></tr>
  <tr><td></td><td>Gilbert</td><td>Gilbert</td></tr>
  <tr><td></td><td>IDOTRR</td><td>Iowa DOT and Rail Road</td></tr>
  <tr><td></td><td>MeadowV</td><td>Meadow Village</td></tr>
  <tr><td></td><td>Mitchel</td><td>Mitchell</td></tr>
  <tr><td></td><td>Names</td><td>North Ames</td></tr>
  <tr><td></td><td>NoRidge</td><td>Northridge</td></tr>
  <tr><td></td><td>NPkVill</td><td>Northpark Villa</td></tr>
  <tr><td></td><td>NridgHt</td><td>Northridge Heights</td></tr>
  <tr><td></td><td>NWAmes</td><td>Northwest Ames</td></tr>
  <tr><td></td><td>OldTown</td><td>Old Town</td></tr>
  <tr><td></td><td>SWISU</td><td>South &amp; West of Iowa State University</td></tr>
  <tr><td></td><td>Sawyer</td><td>Sawyer</td></tr>
  <tr><td></td><td>SawyerW</td><td>Sawyer West</td></tr>
  <tr><td></td><td>Somerst</td><td>Somerset</td></tr>
  <tr><td></td><td>StoneBr</td><td>Stone Brook</td></tr>
  <tr><td></td><td>Timber</td><td>Timberland</td></tr>
  <tr><td></td><td>Veenker</td><td>Veenker</td></tr>

  <!-- Condition1 -->
  <tr>
    <td><b>Condition1</b></td>
    <td colspan="2">Proximity to various conditions.</td>
  </tr>
  <tr><td></td><td>Artery</td><td>Adjacent to arterial street</td></tr>
  <tr><td></td><td>Feedr</td><td>Adjacent to feeder street</td></tr>
  <tr><td></td><td>Norm</td><td>Normal</td></tr>
  <tr><td></td><td>RRNn</td><td>Within 200' of North-South Railroad</td></tr>
  <tr><td></td><td>RRAn</td><td>Adjacent to North-South Railroad</td></tr>
  <tr><td></td><td>PosN</td><td>Near positive off-site feature (e.g., park, greenbelt)</td></tr>
  <tr><td></td><td>PosA</td><td>Adjacent to positive off-site feature</td></tr>
  <tr><td></td><td>RRNe</td><td>Within 200' of East-West Railroad</td></tr>
  <tr><td></td><td>RRAe</td><td>Adjacent to East-West Railroad</td></tr>

  <!-- Condition2 -->
  <tr>
    <td><b>Condition2</b></td>
    <td colspan="2">Proximity to various conditions (if more than one is present).</td>
  </tr>
  <tr><td></td><td>Artery</td><td>Adjacent to arterial street</td></tr>
  <tr><td></td><td>Feedr</td><td>Adjacent to feeder street</td></tr>
  <tr><td></td><td>Norm</td><td>Normal</td></tr>
  <tr><td></td><td>RRNn</td><td>Within 200' of North-South Railroad</td></tr>
  <tr><td></td><td>RRAn</td><td>Adjacent to North-South Railroad</td></tr>
  <tr><td></td><td>PosN</td><td>Near positive off-site feature (e.g., park, greenbelt)</td></tr>
  <tr><td></td><td>PosA</td><td>Adjacent to positive off-site feature</td></tr>
  <tr><td></td><td>RRNe</td><td>Within 200' of East-West Railroad</td></tr>
  <tr><td></td><td>RRAe</td><td>Adjacent to East-West Railroad</td></tr>

  <!-- BldgType -->
  <tr>
    <td><b>BldgType</b></td>
    <td colspan="2">Type of dwelling.</td>
  </tr>
  <tr><td></td><td>1Fam</td><td>Single-family Detached</td></tr>
  <tr><td></td><td>2FmCon</td><td>Two-family Conversion (originally built as one-family dwelling)</td></tr>
  <tr><td></td><td>Duplx</td><td>Duplex</td></tr>
  <tr><td></td><td>TwnhsE</td><td>Townhouse End Unit</td></tr>
  <tr><td></td><td>TwnhsI</td><td>Townhouse Inside Unit</td></tr>

  <!-- HouseStyle -->
  <tr>
    <td><b>HouseStyle</b></td>
    <td colspan="2">Style of dwelling.</td>
  </tr>
  <tr><td></td><td>1Story</td><td>One story</td></tr>
  <tr><td></td><td>1.5Fin</td><td>One and one-half story: 2nd level finished</td></tr>
  <tr><td></td><td>1.5Unf</td><td>One and one-half story: 2nd level unfinished</td></tr>
  <tr><td></td><td>2Story</td><td>Two story</td></tr>
  <tr><td></td><td>2.5Fin</td><td>Two and one-half story: 2nd level finished</td></tr>
  <tr><td></td><td>2.5Unf</td><td>Two and one-half story: 2nd level unfinished</td></tr>
  <tr><td></td><td>SFoyer</td><td>Split Foyer</td></tr>
  <tr><td></td><td>SLvl</td><td>Split Level</td></tr>

  <!-- OverallQual -->
  <tr>
    <td><b>OverallQual</b></td>
    <td colspan="2">Rates the overall material and finish of the house.</td>
  </tr>
  <tr><td></td><td>10</td><td>Very Excellent</td></tr>
  <tr><td></td><td>9</td><td>Excellent</td></tr>
  <tr><td></td><td>8</td><td>Very Good</td></tr>
  <tr><td></td><td>7</td><td>Good</td></tr>
  <tr><td></td><td>6</td><td>Above Average</td></tr>
  <tr><td></td><td>5</td><td>Average</td></tr>
  <tr><td></td><td>4</td><td>Below Average</td></tr>
  <tr><td></td><td>3</td><td>Fair</td></tr>
  <tr><td></td><td>2</td><td>Poor</td></tr>
  <tr><td></td><td>1</td><td>Very Poor</td></tr>

  <!-- OverallCond -->
  <tr>
    <td><b>OverallCond</b></td>
    <td colspan="2">Rates the overall condition of the house.</td>
  </tr>
  <tr><td></td><td>10</td><td>Very Excellent</td></tr>
  <tr><td></td><td>9</td><td>Excellent</td></tr>
  <tr><td></td><td>8</td><td>Very Good</td></tr>
  <tr><td></td><td>7</td><td>Good</td></tr>
  <tr><td></td><td>6</td><td>Above Average</td></tr>
  <tr><td></td><td>5</td><td>Average</td></tr>
  <tr><td></td><td>4</td><td>Below Average</td></tr>
  <tr><td></td><td>3</td><td>Fair</td></tr>
  <tr><td></td><td>2</td><td>Poor</td></tr>
  <tr><td></td><td>1</td><td>Very Poor</td></tr>

  <!-- YearBuilt -->
  <tr>
    <td><b>YearBuilt</b></td>
    <td colspan="2">Original construction date of the house.</td>
  </tr>

  <!-- YearRemodAdd -->
  <tr>
    <td><b>YearRemodAdd</b></td>
    <td colspan="2">Remodel date (same as construction date if no remodeling or additions).</td>
  </tr>

  <!-- RoofStyle -->
  <tr>
    <td><b>RoofStyle</b></td>
    <td colspan="2">Type of roof.</td>
  </tr>
  <tr><td></td><td>Flat</td><td>Flat</td></tr>
  <tr><td></td><td>Gable</td><td>Gable</td></tr>
  <tr><td></td><td>Gambrel</td><td>Gabrel (Barn)</td></tr>
  <tr><td></td><td>Hip</td><td>Hip</td></tr>
  <tr><td></td><td>Mansard</td><td>Mansard</td></tr>
  <tr><td></td><td>Shed</td><td>Shed</td></tr>

  <!-- RoofMatl -->
  <tr>
    <td><b>RoofMatl</b></td>
    <td colspan="2">Roof material.</td>
  </tr>
  <tr><td></td><td>ClyTile</td><td>Clay or Tile</td></tr>
  <tr><td></td><td>CompShg</td><td>Standard (Composite) Shingle</td></tr>
  <tr><td></td><td>Membran</td><td>Membrane</td></tr>
  <tr><td></td><td>Metal</td><td>Metal</td></tr>
  <tr><td></td><td>Roll</td><td>Roll</td></tr>
  <tr><td></td><td>Tar&amp;Grv</td><td>Gravel &amp; Tar</td></tr>
  <tr><td></td><td>WdShake</td><td>Wood Shakes</td></tr>
  <tr><td></td><td>WdShngl</td><td>Wood Shingles</td></tr>

  <!-- Exterior1st -->
  <tr>
    <td><b>Exterior1st</b></td>
    <td colspan="2">Exterior covering on the house.</td>
  </tr>
  <tr><td></td><td>AsbShng</td><td>Asbestos Shingles</td></tr>
  <tr><td></td><td>AsphShn</td><td>Asphalt Shingles</td></tr>
  <tr><td></td><td>BrkComm</td><td>Brick Common</td></tr>
  <tr><td></td><td>BrkFace</td><td>Brick Face</td></tr>
  <tr><td></td><td>CBlock</td><td>Cinder Block</td></tr>
  <tr><td></td><td>CemntBd</td><td>Cement Board</td></tr>
  <tr><td></td><td>HdBoard</td><td>Hard Board</td></tr>
  <tr><td></td><td>ImStucc</td><td>Imitation Stucco</td></tr>
  <tr><td></td><td>MetalSd</td><td>Metal Siding</td></tr>
  <tr><td></td><td>Other</td><td>Other</td></tr>
  <tr><td></td><td>Plywood</td><td>Plywood</td></tr>
  <tr><td></td><td>PreCast</td><td>PreCast</td></tr>
  <tr><td></td><td>Stone</td><td>Stone</td></tr>
  <tr><td></td><td>Stucco</td><td>Stucco</td></tr>
  <tr><td></td><td>VinylSd</td><td>Vinyl Siding</td></tr>
  <tr><td></td><td>Wd Sdng</td><td>Wood Siding</td></tr>
  <tr><td></td><td>WdShing</td><td>Wood Shingles</td></tr>

  <!-- Exterior2nd -->
  <tr>
    <td><b>Exterior2nd</b></td>
    <td colspan="2">Exterior covering on the house (if more than one material).</td>
  </tr>
  <tr><td></td><td>AsbShng</td><td>Asbestos Shingles</td></tr>
  <tr><td></td><td>AsphShn</td><td>Asphalt Shingles</td></tr>
  <tr><td></td><td>BrkComm</td><td>Brick Common</td></tr>
  <tr><td></td><td>BrkFace</td><td>Brick Face</td></tr>
  <tr><td></td><td>CBlock</td><td>Cinder Block</td></tr>
  <tr><td></td><td>CemntBd</td><td>Cement Board</td></tr>
  <tr><td></td><td>HdBoard</td><td>Hard Board</td></tr>
  <tr><td></td><td>ImStucc</td><td>Imitation Stucco</td></tr>
  <tr><td></td><td>MetalSd</td><td>Metal Siding</td></tr>
  <tr><td></td><td>Other</td><td>Other</td></tr>
  <tr><td></td><td>Plywood</td><td>Plywood</td></tr>
  <tr><td></td><td>PreCast</td><td>PreCast</td></tr>
  <tr><td></td><td>Stone</td><td>Stone</td></tr>
  <tr><td></td><td>Stucco</td><td>Stucco</td></tr>
  <tr><td></td><td>VinylSd</td><td>Vinyl Siding</td></tr>
  <tr><td></td><td>Wd Sdng</td><td>Wood Siding</td></tr>
  <tr><td></td><td>WdShing</td><td>Wood Shingles</td></tr>

  <!-- MasVnrType -->
  <tr>
    <td><b>MasVnrType</b></td>
    <td colspan="2">Masonry veneer type.</td>
  </tr>
  <tr><td></td><td>BrkCmn</td><td>Brick Common</td></tr>
  <tr><td></td><td>BrkFace</td><td>Brick Face</td></tr>
  <tr><td></td><td>CBlock</td><td>Cinder Block</td></tr>
  <tr><td></td><td>None</td><td>None</td></tr>
  <tr><td></td><td>Stone</td><td>Stone</td></tr>

  <!-- MasVnrArea -->
  <tr>
    <td><b>MasVnrArea</b></td>
    <td colspan="2">Masonry veneer area in square feet.</td>
  </tr>

  <!-- ExterQual -->
  <tr>
    <td><b>ExterQual</b></td>
    <td colspan="2">Evaluates the quality of the material on the exterior.</td>
  </tr>
  <tr><td></td><td>Ex</td><td>Excellent</td></tr>
  <tr><td></td><td>Gd</td><td>Good</td></tr>
  <tr><td></td><td>TA</td><td>Average/Typical</td></tr>
  <tr><td></td><td>Fa</td><td>Fair</td></tr>
  <tr><td></td><td>Po</td><td>Poor</td></tr>

  <!-- ExterCond -->
  <tr>
    <td><b>ExterCond</b></td>
    <td colspan="2">Evaluates the present condition of the material on the exterior.</td>
  </tr>
  <tr><td></td><td>Ex</td><td>Excellent</td></tr>
  <tr><td></td><td>Gd</td><td>Good</td></tr>
  <tr><td></td><td>TA</td><td>Average/Typical</td></tr>
  <tr><td></td><td>Fa</td><td>Fair</td></tr>
  <tr><td></td><td>Po</td><td>Poor</td></tr>

  <!-- Foundation -->
  <tr>
    <td><b>Foundation</b></td>
    <td colspan="2">Type of foundation.</td>
  </tr>
  <tr><td></td><td>BrkTil</td><td>Brick &amp; Tile</td></tr>
  <tr><td></td><td>CBlock</td><td>Cinder Block</td></tr>
  <tr><td></td><td>PConc</td><td>Poured Concrete</td></tr>
  <tr><td></td><td>Slab</td><td>Slab</td></tr>
  <tr><td></td><td>Stone</td><td>Stone</td></tr>
  <tr><td></td><td>Wood</td><td>Wood</td></tr>

  <!-- BsmtQual -->
  <tr>
    <td><b>BsmtQual</b></td>
    <td colspan="2">Evaluates the height of the basement.</td>
  </tr>
  <tr><td></td><td>Ex</td><td>Excellent (100+ inches)</td></tr>
  <tr><td></td><td>Gd</td><td>Good (90–99 inches)</td></tr>
  <tr><td></td><td>TA</td><td>Typical (80–89 inches)</td></tr>
  <tr><td></td><td>Fa</td><td>Fair (70–79 inches)</td></tr>
  <tr><td></td><td>Po</td><td>Poor (&lt;70 inches)</td></tr>
  <tr><td></td><td>NA</td><td>No Basement</td></tr>

  <!-- BsmtCond -->
  <tr>
    <td><b>BsmtCond</b></td>
    <td colspan="2">Evaluates the general condition of the basement.</td>
  </tr>
  <tr><td></td><td>Ex</td><td>Excellent</td></tr>
  <tr><td></td><td>Gd</td><td>Good</td></tr>
  <tr><td></td><td>TA</td><td>Typical – slight dampness allowed</td></tr>
  <tr><td></td><td>Fa</td><td>Fair – dampness or some cracking or settling</td></tr>
  <tr><td></td><td>Po</td><td>Poor – severe cracking, settling, or wetness</td></tr>
  <tr><td></td><td>NA</td><td>No Basement</td></tr>

  <!-- BsmtExposure -->
  <tr>
    <td><b>BsmtExposure</b></td>
    <td colspan="2">Refers to walkout or garden level walls.</td>
  </tr>
  <tr><td></td><td>Gd</td><td>Good Exposure</td></tr>
  <tr><td></td><td>Av</td><td>Average Exposure (split levels or foyers typically score average or above)</td></tr>
  <tr><td></td><td>Mn</td><td>Minimum Exposure</td></tr>
  <tr><td></td><td>No</td><td>No Exposure</td></tr>
  <tr><td></td><td>NA</td><td>No Basement</td></tr>

  <!-- BsmtFinType1 -->
  <tr>
    <td><b>BsmtFinType1</b></td>
    <td colspan="2">Rating of basement finished area (Type 1).</td>
  </tr>
  <tr><td></td><td>GLQ</td><td>Good Living Quarters</td></tr>
  <tr><td></td><td>ALQ</td><td>Average Living Quarters</td></tr>
  <tr><td></td><td>BLQ</td><td>Below Average Living Quarters</td></tr>
  <tr><td></td><td>Rec</td><td>Average Rec Room</td></tr>
  <tr><td></td><td>LwQ</td><td>Low Quality</td></tr>
  <tr><td></td><td>Unf</td><td>Unfinished</td></tr>
  <tr><td></td><td>NA</td><td>No Basement</td></tr>

  <!-- BsmtFinSF1 -->
  <tr>
    <td><b>BsmtFinSF1</b></td>
    <td colspan="2">Type 1 finished square feet.</td>
  </tr>

  <!-- BsmtFinType2 -->
  <tr>
    <td><b>BsmtFinType2</b></td>
    <td colspan="2">Rating of basement finished area (Type 2, if multiple types).</td>
  </tr>
  <tr><td></td><td>GLQ</td><td>Good Living Quarters</td></tr>
  <tr><td></td><td>ALQ</td><td>Average Living Quarters</td></tr>
  <tr><td></td><td>BLQ</td><td>Below Average Living Quarters</td></tr>
  <tr><td></td><td>Rec</td><td>Average Rec Room</td></tr>
  <tr><td></td><td>LwQ</td><td>Low Quality</td></tr>
  <tr><td></td><td>Unf</td><td>Unfinished</td></tr>
  <tr><td></td><td>NA</td><td>No Basement</td></tr>

  <!-- BsmtFinSF2 -->
  <tr>
    <td><b>BsmtFinSF2</b></td>
    <td colspan="2">Type 2 finished square feet.</td>
  </tr>

  <!-- BsmtUnfSF -->
  <tr>
    <td><b>BsmtUnfSF</b></td>
    <td colspan="2">Unfinished square feet of basement area.</td>
  </tr>

  <!-- TotalBsmtSF -->
  <tr>
    <td><b>TotalBsmtSF</b></td>
    <td colspan="2">Total square feet of basement area.</td>
  </tr>

  <!-- Heating -->
  <tr>
    <td><b>Heating</b></td>
    <td colspan="2">Type of heating.</td>
  </tr>
  <tr><td></td><td>Floor</td><td>Floor Furnace</td></tr>
  <tr><td></td><td>GasA</td><td>Gas forced warm air furnace</td></tr>
  <tr><td></td><td>GasW</td><td>Gas hot water or steam heat</td></tr>
  <tr><td></td><td>Grav</td><td>Gravity furnace</td></tr>
  <tr><td></td><td>OthW</td><td>Hot water or steam heat other than gas</td></tr>
  <tr><td></td><td>Wall</td><td>Wall furnace</td></tr>

  <!-- HeatingQC -->
  <tr>
    <td><b>HeatingQC</b></td>
    <td colspan="2">Heating quality and condition.</td>
  </tr>
  <tr><td></td><td>Ex</td><td>Excellent</td></tr>
  <tr><td></td><td>Gd</td><td>Good</td></tr>
  <tr><td></td><td>TA</td><td>Average/Typical</td></tr>
  <tr><td></td><td>Fa</td><td>Fair</td></tr>
  <tr><td></td><td>Po</td><td>Poor</td></tr>

  <!-- CentralAir -->
  <tr>
    <td><b>CentralAir</b></td>
    <td colspan="2">Central air conditioning.</td>
  </tr>
  <tr><td></td><td>N</td><td>No</td></tr>
  <tr><td></td><td>Y</td><td>Yes</td></tr>

  <!-- Electrical -->
  <tr>
    <td><b>Electrical</b></td>
    <td colspan="2">Electrical system.</td>
  </tr>
  <tr><td></td><td>SBrkr</td><td>Standard Circuit Breakers &amp; Romex</td></tr>
  <tr><td></td><td>FuseA</td><td>Fuse Box over 60 AMP and all Romex wiring (Average)</td></tr>
  <tr><td></td><td>FuseF</td><td>60 AMP Fuse Box and mostly Romex wiring (Fair)</td></tr>
  <tr><td></td><td>FuseP</td><td>60 AMP Fuse Box and mostly knob &amp; tube wiring (Poor)</td></tr>
  <tr><td></td><td>Mix</td><td>Mixed</td></tr>

  <!-- 1stFlrSF -->
  <tr>
    <td><b>1stFlrSF</b></td>
    <td colspan="2">First floor square feet.</td>
  </tr>

  <!-- 2ndFlrSF -->
  <tr>
    <td><b>2ndFlrSF</b></td>
    <td colspan="2">Second floor square feet.</td>
  </tr>

  <!-- LowQualFinSF -->
  <tr>
    <td><b>LowQualFinSF</b></td>
    <td colspan="2">Low quality finished square feet (all floors).</td>
  </tr>

  <!-- GrLivArea -->
  <tr>
    <td><b>GrLivArea</b></td>
    <td colspan="2">Above grade (ground) living area square feet.</td>
  </tr>

  <!-- BsmtFullBath -->
  <tr>
    <td><b>BsmtFullBath</b></td>
    <td colspan="2">Basement full bathrooms.</td>
  </tr>

  <!-- BsmtHalfBath -->
  <tr>
    <td><b>BsmtHalfBath</b></td>
    <td colspan="2">Basement half bathrooms.</td>
  </tr>

  <!-- FullBath -->
  <tr>
    <td><b>FullBath</b></td>
    <td colspan="2">Full bathrooms above grade.</td>
  </tr>

  <!-- HalfBath -->
  <tr>
    <td><b>HalfBath</b></td>
    <td colspan="2">Half bathrooms above grade.</td>
  </tr>

  <!-- Bedroom -->
  <tr>
    <td><b>Bedroom</b></td>
    <td colspan="2">Bedrooms above grade (does NOT include basement bedrooms).</td>
  </tr>

  <!-- Kitchen -->
  <tr>
    <td><b>Kitchen</b></td>
    <td colspan="2">Kitchens above grade.</td>
  </tr>

  <!-- TotRmsAbvGrd -->
  <tr>
    <td><b>TotRmsAbvGrd</b></td>
    <td colspan="2">Total rooms above grade (does NOT include bathrooms).</td>
  </tr>

  <!-- KitchenQual -->
  <tr>
    <td><b>KitchenQual</b></td>
    <td colspan="2">Kitchen quality.</td>
  </tr>
  <tr><td></td><td>Ex</td><td>Excellent</td></tr>
  <tr><td></td><td>Gd</td><td>Good</td></tr>
  <tr><td></td><td>TA</td><td>Typical/Average</td></tr>
  <tr><td></td><td>Fa</td><td>Fair</td></tr>
  <tr><td></td><td>Po</td><td>Poor</td></tr>

  <!-- Functional -->
  <tr>
    <td><b>Functional</b></td>
    <td colspan="2">Home functionality (assume typical unless deductions are warranted).</td>
  </tr>
  <tr><td></td><td>Typ</td><td>Typical Functionality</td></tr>
  <tr><td></td><td>Min1</td><td>Minor Deductions 1</td></tr>
  <tr><td></td><td>Min2</td><td>Minor Deductions 2</td></tr>
  <tr><td></td><td>Mod</td><td>Moderate Deductions</td></tr>
  <tr><td></td><td>Maj1</td><td>Major Deductions 1</td></tr>
  <tr><td></td><td>Maj2</td><td>Major Deductions 2</td></tr>
  <tr><td></td><td>Sev</td><td>Severely Damaged</td></tr>
  <tr><td></td><td>Sal</td><td>Salvage Only</td></tr>

  <!-- Fireplaces -->
  <tr>
    <td><b>Fireplaces</b></td>
    <td colspan="2">Number of fireplaces in the house.</td>
  </tr>

  <!-- FireplaceQu -->
  <tr>
    <td><b>FireplaceQu</b></td>
    <td colspan="2">Fireplace quality.</td>
  </tr>
  <tr><td></td><td>Ex</td><td>Excellent – Exceptional Masonry Fireplace</td></tr>
  <tr><td></td><td>Gd</td><td>Good – Masonry Fireplace in main level</td></tr>
  <tr><td></td><td>TA</td><td>Average – Prefabricated Fireplace in main living area or Masonry Fireplace in basement</td></tr>
  <tr><td></td><td>Fa</td><td>Fair – Prefabricated Fireplace in basement</td></tr>
  <tr><td></td><td>Po</td><td>Poor – Ben Franklin Stove</td></tr>
  <tr><td></td><td>NA</td><td>No Fireplace</td></tr>

  <!-- GarageType -->
  <tr>
    <td><b>GarageType</b></td>
    <td colspan="2">Garage location relative to the house.</td>
  </tr>
  <tr><td></td><td>2Types</td><td>More than one type of garage</td></tr>
  <tr><td></td><td>Attchd</td><td>Attached to home</td></tr>
  <tr><td></td><td>Basment</td><td>Basement Garage</td></tr>
  <tr><td></td><td>BuiltIn</td><td>Built-In (Garage part of house - typically has room above garage)</td></tr>
  <tr><td></td><td>CarPort</td><td>Car Port</td></tr>
  <tr><td></td><td>Detchd</td><td>Detached from home</td></tr>
  <tr><td></td><td>NA</td><td>No Garage</td></tr>

  <!-- GarageYrBlt -->
  <tr>
    <td><b>GarageYrBlt</b></td>
    <td colspan="2">Year the garage was built.</td>
  </tr>

  <!-- GarageFinish -->
  <tr>
    <td><b>GarageFinish</b></td>
    <td colspan="2">Interior finish of the garage.</td>
  </tr>
  <tr><td></td><td>Fin</td><td>Finished</td></tr>
  <tr><td></td><td>RFn</td><td>Rough Finished</td></tr>
  <tr><td></td><td>Unf</td><td>Unfinished</td></tr>
  <tr><td></td><td>NA</td><td>No Garage</td></tr>

  <!-- GarageCars -->
  <tr>
    <td><b>GarageCars</b></td>
    <td colspan="2">Size of garage in car capacity.</td>
  </tr>

  <!-- GarageArea -->
  <tr>
    <td><b>GarageArea</b></td>
    <td colspan="2">Size of garage in square feet.</td>
  </tr>

  <!-- GarageQual -->
  <tr>
    <td><b>GarageQual</b></td>
    <td colspan="2">Garage quality.</td>
  </tr>
  <tr><td></td><td>Ex</td><td>Excellent</td></tr>
  <tr><td></td><td>Gd</td><td>Good</td></tr>
  <tr><td></td><td>TA</td><td>Typical/Average</td></tr>
  <tr><td></td><td>Fa</td><td>Fair</td></tr>
  <tr><td></td><td>Po</td><td>Poor</td></tr>
  <tr><td></td><td>NA</td><td>No Garage</td></tr>

  <!-- GarageCond -->
  <tr>
    <td><b>GarageCond</b></td>
    <td colspan="2">Garage condition.</td>
  </tr>
  <tr><td></td><td>Ex</td><td>Excellent</td></tr>
  <tr><td></td><td>Gd</td><td>Good</td></tr>
  <tr><td></td><td>TA</td><td>Typical/Average</td></tr>
  <tr><td></td><td>Fa</td><td>Fair</td></tr>
  <tr><td></td><td>Po</td><td>Poor</td></tr>
  <tr><td></td><td>NA</td><td>No Garage</td></tr>

  <!-- PavedDrive -->
  <tr>
    <td><b>PavedDrive</b></td>
    <td colspan="2">Paved driveway.</td>
  </tr>
  <tr><td></td><td>Y</td><td>Paved</td></tr>
  <tr><td></td><td>P</td><td>Partial Pavement</td></tr>
  <tr><td></td><td>N</td><td>Dirt/Gravel</td></tr>

  <!-- WoodDeckSF -->
  <tr>
    <td><b>WoodDeckSF</b></td>
    <td colspan="2">Wood deck area in square feet.</td>
  </tr>

  <!-- OpenPorchSF -->
  <tr>
    <td><b>OpenPorchSF</b></td>
    <td colspan="2">Open porch area in square feet.</td>
  </tr>

  <!-- EnclosedPorch -->
  <tr>
    <td><b>EnclosedPorch</b></td>
    <td colspan="2">Enclosed porch area in square feet.</td>
  </tr>

  <!-- 3SsnPorch -->
  <tr>
    <td><b>3SsnPorch</b></td>
    <td colspan="2">Three season porch area in square feet.</td>
  </tr>

  <!-- ScreenPorch -->
  <tr>
    <td><b>ScreenPorch</b></td>
    <td colspan="2">Screen porch area in square feet.</td>
  </tr>

  <!-- PoolArea -->
  <tr>
    <td><b>PoolArea</b></td>
    <td colspan="2">Pool area in square feet.</td>
  </tr>

  <!-- PoolQC -->
  <tr>
    <td><b>PoolQC</b></td>
    <td colspan="2">Pool quality.</td>
  </tr>
  <tr><td></td><td>Ex</td><td>Excellent</td></tr>
  <tr><td></td><td>Gd</td><td>Good</td></tr>
  <tr><td></td><td>TA</td><td>Average/Typical</td></tr>
  <tr><td></td><td>Fa</td><td>Fair</td></tr>
  <tr><td></td><td>NA</td><td>No Pool</td></tr>

  <!-- Fence -->
  <tr>
    <td><b>Fence</b></td>
    <td colspan="2">Fence quality.</td>
  </tr>
  <tr><td></td><td>GdPrv</td><td>Good Privacy</td></tr>
  <tr><td></td><td>MnPrv</td><td>Minimum Privacy</td></tr>
  <tr><td></td><td>GdWo</td><td>Good Wood</td></tr>
  <tr><td></td><td>MnWw</td><td>Minimum Wood/Wire</td></tr>
  <tr><td></td><td>NA</td><td>No Fence</td></tr>

  <!-- MiscFeature -->
  <tr>
    <td><b>MiscFeature</b></td>
    <td colspan="2">Miscellaneous feature not covered in other categories.</td>
  </tr>
  <tr><td></td><td>Elev</td><td>Elevator</td></tr>
  <tr><td></td><td>Gar2</td><td>2nd Garage (if not described in garage section)</td></tr>
  <tr><td></td><td>Othr</td><td>Other</td></tr>
  <tr><td></td><td>Shed</td><td>Shed (over 100 SF)</td></tr>
  <tr><td></td><td>TenC</td><td>Tennis Court</td></tr>
  <tr><td></td><td>NA</td><td>None</td></tr>

  <!-- MiscVal -->
  <tr>
    <td><b>MiscVal</b></td>
    <td colspan="2">Dollar value of miscellaneous feature.</td>
  </tr>

  <!-- MoSold -->
  <tr>
    <td><b>MoSold</b></td>
    <td colspan="2">Month Sold (MM).</td>
  </tr>

  <!-- YrSold -->
  <tr>
    <td><b>YrSold</b></td>
    <td colspan="2">Year Sold (YYYY).</td>
  </tr>

  <!-- SaleType -->
  <tr>
    <td><b>SaleType</b></td>
    <td colspan="2">Type of sale.</td>
  </tr>
  <tr><td></td><td>WD</td><td>Warranty Deed - Conventional</td></tr>
  <tr><td></td><td>CWD</td><td>Warranty Deed - Cash</td></tr>
  <tr><td></td><td>VWD</td><td>Warranty Deed - VA Loan</td></tr>
  <tr><td></td><td>New</td><td>Home just constructed and sold</td></tr>
  <tr><td></td><td>COD</td><td>Court Officer Deed/Estate</td></tr>
  <tr><td></td><td>Con</td><td>Contract 15% Down Payment Regular Terms</td></tr>
  <tr><td></td><td>ConLw</td><td>Contract Low Down Payment and Low Interest</td></tr>
  <tr><td></td><td>ConLI</td><td>Contract Low Interest</td></tr>
  <tr><td></td><td>ConLD</td><td>Contract Low Down</td></tr>
  <tr><td></td><td>Oth</td><td>Other</td></tr>

  <!-- SaleCondition -->
  <tr>
    <td><b>SaleCondition</b></td>
    <td colspan="2">Condition of sale.</td>
  </tr>
  <tr><td></td><td>Normal</td><td>Normal Sale</td></tr>
  <tr><td></td><td>Abnorml</td><td>Abnormal Sale – trade, foreclosure, short sale</td></tr>
  <tr><td></td><td>AdjLand</td><td>Adjoining Land Purchase</td></tr>
  <tr><td></td><td>Alloca</td><td>Allocation – two linked properties with separate deeds (e.g., condo with a garage unit)</td></tr>
  <tr><td></td><td>Family</td><td>Sale between family members</td></tr>
  <tr><td></td><td>Partial</td><td>Home was not completed when last assessed (associated with New Homes)</td></tr>
</table>

</details>

## Steps Taken

### Data Loading
Unzipped the Kaggle dataset and loaded `train.csv` and `test.csv` into pandas Dataframes 


### Data Preprocessing 

All missing numeric values in `df_train` and `df_test` were filled with median of each column from `df_train`. The median is used as it is not as sensitive to outliers compared to the mean. This ensures no missing values remain, avoiding training errors. 

Categorical columns are one-hot encoded using `pd.get_dummies`, which converts text features into numeric form while avoiding multicollinearity. The encoded train and test sets were them aligned to ensure consistent feature columns. The target variable `SalePrice` is then log-transformed to reduce skewness stabilize variance, improving the model's ability to learn linear relationships. 

All features were standardized with `StandardScaler()` so they share the same scale, preventing large-valued variables from dominating the regression model. 


Finally, the dataset was split into training (70%) and validation (30%) sets with `random_state=42` to ensure reproducability, allowing model performance to be evaluated before applying it to the unseen Kaggle test set. 


### Model Training and Evaluation
* Implemented 2 models:
   * Linear Regression
   * Ridge Regression


#### Linear Regression

Model performance was evaluated using 5-fold cross-validation (`cv=5`) with negative mean squared error (MSE) to evaluare model stability. The root mean square error (RMSE) was then calculated for each fold to measure the average prediction error, where lower values indicate a better fit. 

The results were `Linear Regression CV RMSE: [0.12892002, 0.28797436, 0.16607483, 0.11685145, 0.21874038]` with a mean of approximately `0.18371220951259662`. 

This suggests that the model's predictions are within about ±20% of the actual house prices performing well for fold (`0.12892002`) but struggling with fold (`0.28797436`). This could indicate that there are outliers or different distributions in that fold. 

The R square score was calculated to measure how much of the variance in the target variable the model could explain. It returns a value of `0.7832441336404377`, meaning the model accounts for approximately 78.3% of the variance in house prices. However, there is still a remaining 21.7% of the variation unexplained, likely due to some factors the model did not capture. This includes missing features, noise and nonlinear patterns. 

   
#### Ridge Regression

Used multiple alpha values to test different regularization strengths, to find the optimal balance between bias and variance. Smaller alphas behave like plain Linear Regression, while larger alpha apply stronger penalties. The best alpha was determined automatically using 5-fold cross-validation with MSE as the evaluation metric:
  
```sh
Best alpha for Ridge: 100.0
```

This means that the alpha value 100 achieved the lowest average MSE. The model's performance across folds was then assessed using RMSE, where lower values indicate better predictive accuracy. 

```sh
Ridge CV RMSE scores: [0.12418077 0.25096448 0.1566775  0.11952967 0.17385427]
Ridge CV RMSE mean: 0.16504133909005891
```

This shows that the model's predictions were within about ±15% of the actual house prices, showing consistent and improved performance compared to Linear Regression. The model fits well for fold (`0.11952967`) and struggled with fold (`0.25096448`). This could indicate outliers or different distributions in that fold.

When validated on unseen data, Ridge Regression achieved an R square value of `0.8794372046130793`. This explains aproximately 87.9% of the variance in house prices. However, there is still a remaining 12.1% of the variation still unexplained, likely due to factors the model did not capture. This includes missing features, noise and nonlinear patterns. 


#### Model Comparison
   
Comparing the two R squared values, the R squared value from Ridge Regression is higher than that of Linear Regression. This shows that the Ridge model explains a greater proportion of the variance in house prices and demonstrates better generalization on unseen data. The improvement can be attributed to L2 regularization, which penalizes overly large coefficients, reducing overfitting and stabilizing the model's predictions without significantly increasing bias.

#### Ridge Alpha Experiments

To evaluate the effect of regularization strength, multiple alpha values were tested sequentially using 5-fold cross-validation. The code was looped through each alpha, fitting a Ridge Regression model and computing the validation RMSE after reversing the log transformation with`np.expm1()`. 

The results showed that as alpha increased from 0.1 to 10, the validation RMSE steadily decreased from $24,379.52 to $24,899.69, indicating that alpha 10 achieved the best generalization among tested values. Increasing alpha beyond this point caused higher RMSE, likely due to underfitting as excessive regularization oversmooths the model and reduces predictive accuracy. 

The final model used the best alpha value 10, applied to the unseen test data to generate predictions. The trained `ridge_cv` model first produced predictions in log-transformed form `y_test_pred_log`, which were then converted back to the original dollar scale using `np.expm1()`. This provided the model's final predicted house prices on the test dataset, ready for performance evaluation.

### Final Model Evaluation and Visualization 

#### Correlation heatmap

We computed the correlation matrix of all variables, and the 20 most correlated features with `SalePrice` were selected. A heatmap (Figure 1) was then plotted to visualise how these features relate to one another. This allowed us to identify the impact of different characteristics on house value (`SalePrice`).

<img width="1035" height="800" alt="image" src="https://github.com/user-attachments/assets/ee8d00f8-f8a6-4682-8123-864f36a742e7" />

Figure 1

From Figure 1, factors related to quality (`OverallQual`, `KitchenQual`, `GarageFinish`) and size (`GrLivArea`, `TotalBsmtSF`) exhibit strong positive correlations with price; while factors such materials and finishes (`ExterQual_TA`, `KitchenQual_TA`) have negative or weaker correlations. 

#### Linear Regression Top 20 Coefficients 

<img width="1119" height="547" alt="image" src="https://github.com/user-attachments/assets/88cadf38-f7e2-4266-8ea7-a7bc24e1e67f" />

Figure 2

FIgure 2 shows a barplot that ranks the 20 top features by their absolute coefficient magnitudes for linear regression.

From this, we can conclude that more significant factors impacting sale price of house are factors like Roof and MS zoning(`RoofMatl_CompShg` and `MSzoning_RL`)- we can conclude that they are primary determinants of higher prices. Conversely, factors like ground living area (`GrLivArea`) have comparably smaller coefficients, indicating a smaller influence on price. However, compared to Ridge regression, the coefficient of these parameters could be large, which have extreme weight of overall prediction.

There is also an observable gradual decrease in coefficient sizes. This shows that the model spreads importance more evenly across features and does not rely heavily on one or two. This means that the Ridge Regression prevents overfitting, and is a balanced, reliable model.

#### Ridge Regression Top 20 Coefficients 

<img width="1139" height="547" alt="image" src="https://github.com/user-attachments/assets/ef717ac5-6291-47d2-864c-af123315cde0" />

Figure 3

FIgure 3 shows a barplot that ranks the 20 top features by their absolute coefficient magnitudes for ridge regression.

From this, we can conclude that more significant factors impacting sale price of house are factors like overall material and quality and above-ground living area (`Overallqual` and `GrLivArea`)- we can conclude that they are primary determinants of higher prices. Conversely, factors like House age (`YearBuilt`) have comparably smaller coefficients, indicating a smaller influence on price.

There is also an observable gradual decrease in coefficient sizes. This shows that the model spreads importance more evenly across features and does not rely heavily on one or two. This means that the Ridge Regression prevents overfitting, and is a balanced, reliable model.

#### Scatter plots 
<img width="694" height="552" alt="image" src="https://github.com/user-attachments/assets/bff6add4-a3fe-4f6d-bba3-656cc9f5d418" />
<img width="694" height="552" alt="image" src="https://github.com/user-attachments/assets/fe42d1ca-d265-41b8-ac08-8d3bc6fb1fa4" />


Figure 4 & 5

Figure 4 and 5 show scatter plots for Linear Regression and Ridge Regression respectively. Generally, the predicted line is fitted in the trend of the points, proving the general linear relationship for both. 

## Contributions

Frank: selected dataset, code, script and production of presentation

Lynn: selected dataset, readme file, script and production of presentation

Cadence: selected dataset, readme file, script and production of presentation


   
