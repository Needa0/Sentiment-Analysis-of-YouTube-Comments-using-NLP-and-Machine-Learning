"""YouTube comment collection via the YouTube Data API v3.

Implements the "Dataset Collection" phase of the Plan of Work:

    "Configure YouTube Data API using Google API Client library and collect
     multilingual YouTube comments. Store collected comments in structured CSV
     format using Pandas."

The PPT ("System Architecture") additionally specifies that the YouTube API
component "Fetches comments by video ID or channel".

A YouTube Data API key is required for live collection and is read from the
``YOUTUBE_API_KEY`` environment variable (or passed explicitly). When no key /
network is available, :func:`load_comments_csv` lets the rest of the pipeline run
on a previously collected / sample CSV so the system remains fully verifiable.
"""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd

CSV_COLUMNS = ["comment_id", "video_id", "author", "comment", "like_count"]


class YouTubeCommentCollector:
    """Collect comments from YouTube using the Google API Client library."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
        self._service = None

    def _get_service(self):
        """Lazily build the YouTube Data API v3 service object.

        Imported lazily so the rest of the pipeline does not require the
        Google API Client to be importable at module import time.
        """
        if self._service is not None:
            return self._service
        if not self.api_key:
            raise ValueError(
                "A YouTube Data API key is required. Set the YOUTUBE_API_KEY "
                "environment variable or pass api_key=... ."
            )
        from googleapiclient.discovery import build

        self._service = build("youtube", "v3", developerKey=self.api_key)
        return self._service

    def fetch_video_comments(
        self, video_id: str, max_comments: int = 200
    ) -> List[dict]:
        """Fetch top-level comments for a single video id."""
        service = self._get_service()
        comments: List[dict] = []
        request = service.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments),
            textFormat="plainText",
        )
        while request is not None and len(comments) < max_comments:
            response = request.execute()
            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append(
                    {
                        "comment_id": item["id"],
                        "video_id": video_id,
                        "author": snippet.get("authorDisplayName", ""),
                        "comment": snippet.get("textDisplay", ""),
                        "like_count": snippet.get("likeCount", 0),
                    }
                )
            request = service.commentThreads().list_next(request, response)
        return comments[:max_comments]

    def fetch_channel_comments(
        self, channel_id: str, max_videos: int = 5, max_comments_per_video: int = 100
    ) -> List[dict]:
        """Fetch comments across the most recent videos of a channel."""
        service = self._get_service()
        search_response = (
            service.search()
            .list(
                part="id",
                channelId=channel_id,
                maxResults=min(50, max_videos),
                order="date",
                type="video",
            )
            .execute()
        )
        comments: List[dict] = []
        for item in search_response.get("items", []):
            video_id = item["id"]["videoId"]
            comments.extend(
                self.fetch_video_comments(video_id, max_comments_per_video)
            )
        return comments

    def collect_to_csv(
        self,
        output_path: str,
        video_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        max_comments: int = 200,
    ) -> pd.DataFrame:
        """Collect comments by video id or channel id and store them as CSV."""
        if not video_id and not channel_id:
            raise ValueError("Provide either video_id or channel_id.")
        if video_id:
            rows = self.fetch_video_comments(video_id, max_comments)
        else:
            rows = self.fetch_channel_comments(channel_id)
        df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        save_comments_csv(df, output_path)
        return df


def save_comments_csv(df: pd.DataFrame, path: str) -> None:
    """Persist a comments DataFrame to CSV using Pandas."""
    df.to_csv(path, index=False, encoding="utf-8")


def load_comments_csv(path: str) -> pd.DataFrame:
    """Load a previously collected comments CSV using Pandas."""
    return pd.read_csv(path, encoding="utf-8")
