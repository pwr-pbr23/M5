import javascript
import Analyze

from AnalyzedExtAPString accessPath, ExtEventString eventName
where knownUnknown(accessPath, eventName)
select accessPath, eventName