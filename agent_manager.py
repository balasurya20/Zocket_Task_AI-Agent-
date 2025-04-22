import os
from scraper_agent import ScraperAgent
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentManager:
    
    def __init__(self, hf_api_key=None):
        try:
            self.hf_api_key = hf_api_key or os.environ.get("HF_API_KEY")
            self.scraper_agent = ScraperAgent(hf_api_key=self.hf_api_key)
            logger.info("Agent manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent manager: {str(e)}")
            raise

    def validate_url(self, url):
        url_pattern = re.compile(
            r'^(https?://)?(www\.)?'
            r'([a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)+)'
            r'(/[^\s]*)?$'
        )

        if not url_pattern.match(url):
            return False, "Invalid URL format. Please enter a valid URL."

        return True, url

    def process_website(self, url):
        try:
            is_valid, url_or_error = self.validate_url(url)
            if not is_valid:
                return {"success": False, "error": url_or_error}

            logger.info(f"Processing URL: {url}")
            result = self.scraper_agent.process_url(url)

            if result["success"]:
                logger.info(f"Successfully processed URL: {url}")
            else:
                logger.error(f"Failed to process URL: {url}, Error: {result.get('error', 'Unknown error')}")

            return result

        except Exception as e:
            logger.error(f"Error in process_website: {str(e)}")
            return {"success": False, "error": f"Processing failed: {str(e)}"}

    def get_health_status(self):
        status = {
            "agent_manager": "healthy",
            "scraper_agent": "unknown",
            "api_key": "not set" if not self.hf_api_key else "set"
        }

        try:
            if hasattr(self.scraper_agent, 'scrape_website'):
                status["scraper_agent"] = "healthy"
            else:
                status["scraper_agent"] = "not properly initialized"
        except Exception as e:
            status["scraper_agent"] = f"error: {str(e)}"

        return status
